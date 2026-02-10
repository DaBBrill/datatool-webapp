"""
Streamlit Vendor Inventory Analysis Application

This application connects to a PostgreSQL database and provides:
- Vendor-centric inventory analysis view
- Purchase history by year for selected vendor
- Opportunity and oversupply calculations

Database Tables Required:
- stock_price: Product inventory (id, sku, description, easton_on_hand, total_on_hand, base_price, created_at, updated_at)
- transactions: Sales transactions (id, invoice_date, invoice_number, sku, quantity, sales_amount, margin, customer_code, customer_name, vendor, created_at)

Environment Variables Required:
- DATABASE_URL: Full PostgreSQL connection string
  OR individual components:
  - DB_HOST: Database host
  - DB_PORT: Database port
  - DB_NAME: Database name
  - DB_USER: Database user
  - DB_PASSWORD: Database password
"""

import streamlit as st
import psycopg2
import pandas as pd
import os
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlparse
from datetime import datetime
from rapidfuzz import fuzz, process
from streamlit_searchbox import st_searchbox


# Page configuration
st.set_page_config(
    page_title="Vendor Inventory Analysis",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def get_database_connection():
    """
    Establish and cache database connection.
    
    Uses DATABASE_URL if available, otherwise falls back to individual
    connection parameters from environment variables.
    
    Returns:
        psycopg2.connection: Database connection object
    
    Raises:
        Exception: If connection fails or environment variables are missing
    """
    try:
        # First, try to use DATABASE_URL
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Parse the DATABASE_URL
            result = urlparse(database_url)
            conn = psycopg2.connect(
                host=result.hostname,
                port=result.port,
                database=result.path[1:],  # Remove leading '/'
                user=result.username,
                password=result.password
            )
        else:
            # Fall back to individual environment variables
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )
        
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.info("Please ensure DATABASE_URL or DB_* environment variables are properly configured.")
        raise


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_vendors(_conn, only_2025: bool = True, sales_reps: List[str] = None,
                states: List[str] = None, dealer_terms: List[str] = None,
                statuses: List[str] = None) -> List[str]:
    """
    Retrieve list of unique vendor names from transactions table.
    Normalizes vendor names to title case for consistent display.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        only_2025: If True, only show vendors with 2025 sales (default: True)
        sales_reps: List of sales representatives to filter by (optional)
        states: List of states (billing address) to filter by (optional)
        dealer_terms: List of dealer terms to filter by (optional)
        statuses: List of statuses to filter by (optional, 'Active' or 'Inactive')
    
    Returns:
        List[str]: Sorted list of unique normalized vendor names
    """
    try:
        cursor = _conn.cursor()
        
        # Build WHERE clause conditions
        where_conditions = ["t.vendor IS NOT NULL"]
        params = []
        
        # Add 2025 filter
        if only_2025:
            where_conditions.append("EXTRACT(YEAR FROM t.invoice_date) = 2025")
        
        # Determine if we need to join with vendor_metadata
        needs_metadata_join = (sales_reps or states or dealer_terms or statuses)
        
        if needs_metadata_join:
            # Build query with vendor_metadata join
            query_parts = [
                "SELECT DISTINCT INITCAP(LOWER(t.vendor)) AS normalized_vendor",
                "FROM transactions t",
                "INNER JOIN vendor_metadata vm ON t.internal_id = vm.internal_id",
                "WHERE " + " AND ".join(where_conditions)
            ]
            
            # Add filter conditions
            if sales_reps:
                placeholders = ','.join(['%s'] * len(sales_reps))
                query_parts.append(f"AND vm.sales_rep IN ({placeholders})")
                params.extend(sales_reps)
            
            if states:
                placeholders = ','.join(['%s'] * len(states))
                query_parts.append(f"AND vm.billing_state_province IN ({placeholders})")
                params.extend(states)
            
            if dealer_terms:
                placeholders = ','.join(['%s'] * len(dealer_terms))
                query_parts.append(f"AND vm.dealer_terms IN ({placeholders})")
                params.extend(dealer_terms)
            
            if statuses:
                # Convert status labels to database values
                status_conditions = []
                for status in statuses:
                    if status == "Active":
                        status_conditions.append("(vm.inactive IS NULL OR vm.inactive NOT IN ('yes', 'y', 'true', '1', 'Yes', 'Y', 'True', 'TRUE'))")
                    elif status == "Inactive":
                        status_conditions.append("(vm.inactive IN ('yes', 'y', 'true', '1', 'Yes', 'Y', 'True', 'TRUE'))")
                
                if status_conditions:
                    query_parts.append(f"AND ({' OR '.join(status_conditions)})")
            
            query_parts.append("ORDER BY normalized_vendor;")
            query = "\n".join(query_parts)
        else:
            # Simple query without metadata join
            query = f"""
                SELECT DISTINCT INITCAP(LOWER(vendor)) AS normalized_vendor
                FROM transactions t
                WHERE {' AND '.join(where_conditions)}
                ORDER BY normalized_vendor;
            """
        
        cursor.execute(query, params if params else None)
        vendors = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return vendors
    except Exception as e:
        st.error(f"Failed to retrieve vendor list: {str(e)}")
        return []


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_vendor_inventory_analysis(_conn, vendor_name: str) -> Optional[pd.DataFrame]:
    """
    Get inventory analysis for a specific vendor.
    Normalizes vendor names for grouping to handle inconsistent casing.
    
    This function:
    1. Gets all SKUs the vendor sold in 2025
    2. Aggregates vendor sales quantities by year (2015-2025)
    3. Gets total 2025 sales across all customers
    4. Joins with stock_price for descriptions, on_hand, and base_price
    5. Calculates Opportunity and Total Oversupply
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        vendor_name: Name of the vendor to analyze (title case)
    
    Returns:
        pd.DataFrame: Inventory analysis data
    """
    try:
        query = """
        WITH vendor_2025_skus AS (
            -- Get SKUs that this vendor sold in 2025
            SELECT DISTINCT sku
            FROM transactions
            WHERE INITCAP(LOWER(vendor)) = %s
              AND EXTRACT(YEAR FROM invoice_date) = 2025
        ),
        vendor_purchase_history AS (
            -- Get vendor's sales history by year for their 2025 SKUs
            SELECT
                t.sku,
                EXTRACT(YEAR FROM t.invoice_date) AS year,
                SUM(t.quantity) AS quantity
            FROM transactions t
            INNER JOIN vendor_2025_skus v2025 ON t.sku = v2025.sku
            WHERE INITCAP(LOWER(t.vendor)) = %s
            GROUP BY t.sku, EXTRACT(YEAR FROM t.invoice_date)
        ),
        all_customers_2025_sales AS (
            -- Get total 2025 sales across all customers for these SKUs
            SELECT
                t.sku,
                SUM(t.quantity) AS total_2025_quantity
            FROM transactions t
            INNER JOIN vendor_2025_skus v2025 ON t.sku = v2025.sku
            WHERE EXTRACT(YEAR FROM t.invoice_date) = 2025
            GROUP BY t.sku
        ),
        pivoted_vendor_data AS (
            -- Pivot vendor sales history by year
            SELECT
                sku,
                SUM(CASE WHEN year = 2015 THEN quantity ELSE 0 END) AS year_2015,
                SUM(CASE WHEN year = 2016 THEN quantity ELSE 0 END) AS year_2016,
                SUM(CASE WHEN year = 2017 THEN quantity ELSE 0 END) AS year_2017,
                SUM(CASE WHEN year = 2018 THEN quantity ELSE 0 END) AS year_2018,
                SUM(CASE WHEN year = 2024 THEN quantity ELSE 0 END) AS year_2024,
                SUM(CASE WHEN year = 2025 THEN quantity ELSE 0 END) AS year_2025
            FROM vendor_purchase_history
            GROUP BY sku
        )
        SELECT
            sp.sku,
            sp.description AS "Name",
            COALESCE(pvd.year_2015, 0) AS "2015",
            COALESCE(pvd.year_2016, 0) AS "2016",
            COALESCE(pvd.year_2017, 0) AS "2017",
            COALESCE(pvd.year_2018, 0) AS "2018",
            COALESCE(pvd.year_2024, 0) AS "2024",
            COALESCE(pvd.year_2025, 0) AS "2025",
            sp.easton_on_hand AS "On_Hand",
            COALESCE(acs.total_2025_quantity, 0) AS "2025 Sales (all customers)",
            sp.base_price AS "Base Price",
            (sp.easton_on_hand - COALESCE(acs.total_2025_quantity, 0)) * sp.base_price AS "Opportunity",
            sp.easton_on_hand - COALESCE(acs.total_2025_quantity, 0) AS "Total Oversupply"
        FROM stock_price sp
        INNER JOIN pivoted_vendor_data pvd ON sp.sku = pvd.sku
        LEFT JOIN all_customers_2025_sales acs ON sp.sku = acs.sku
        ORDER BY sp.sku;
        """
        
        df = pd.read_sql_query(query, _conn, params=(vendor_name, vendor_name))
        return df
        
    except Exception as e:
        st.error(f"Failed to retrieve inventory analysis: {str(e)}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_vendor_metadata_by_internal_id(_conn, internal_id: str) -> Optional[Dict]:
    """
    Fetch vendor metadata from vendor_metadata table by internal_id.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        internal_id: The internal_id to lookup
    
    Returns:
        Dict with vendor metadata fields, or None if not found
    """
    try:
        cursor = _conn.cursor()
        query = """
            SELECT
                internal_id,
                name,
                sales_rep,
                billing_address_1,
                billing_address_2,
                billing_city,
                billing_state_province,
                billing_zip,
                date_of_last_sale,
                inactive,
                dealer_terms
            FROM vendor_metadata
            WHERE internal_id = %s;
        """
        cursor.execute(query, (internal_id,))
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return {
                'internal_id': row[0],
                'name': row[1],
                'sales_rep': row[2],
                'billing_address_1': row[3],
                'billing_address_2': row[4],
                'billing_city': row[5],
                'billing_state_province': row[6],
                'billing_zip': row[7],
                'date_of_last_sale': row[8],
                'inactive': row[9],
                'dealer_terms': row[10]
            }
        return None
    except Exception as e:
        st.error(f"Failed to fetch vendor metadata: {str(e)}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_vendor_most_recent_transaction_date(_conn, internal_id: str) -> Optional[datetime]:
    """
    Get the most recent transaction date for a vendor by internal_id.
    Used as fallback when date_of_last_sale is not available in vendor_metadata.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        internal_id: The internal_id to lookup
    
    Returns:
        datetime of most recent transaction, or None if no transactions found
    """
    try:
        cursor = _conn.cursor()
        query = """
            SELECT MAX(invoice_date) as most_recent_date
            FROM transactions
            WHERE internal_id = %s
              AND invoice_date IS NOT NULL;
        """
        cursor.execute(query, (internal_id,))
        row = cursor.fetchone()
        cursor.close()
        
        if row and row[0]:
            return row[0]
        return None
    except Exception as e:
        st.error(f"Failed to fetch most recent transaction date: {str(e)}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_vendor_internal_id(_conn, vendor_name: str) -> Optional[str]:
    """
    Check if a vendor name has an internal_id mapping in transactions table.
    Uses normalized vendor name matching. If multiple internal_ids exist,
    returns the most common one.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        vendor_name: Vendor name to check (title case)
    
    Returns:
        internal_id if found, None otherwise
    """
    try:
        cursor = _conn.cursor()
        query = """
            SELECT internal_id, COUNT(*) as count
            FROM transactions
            WHERE INITCAP(LOWER(vendor)) = INITCAP(LOWER(%s))
              AND internal_id IS NOT NULL
            GROUP BY internal_id
            ORDER BY count DESC
            LIMIT 1;
        """
        cursor.execute(query, (vendor_name,))
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return row[0]
        return None
    except Exception as e:
        st.error(f"Failed to check vendor mapping: {str(e)}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_all_vendor_metadata_names(_conn) -> List[Tuple[str, str]]:
    """
    Fetch all vendor names and internal_ids from vendor_metadata table.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
    
    Returns:
        List of tuples (internal_id, name)
    """
    try:
        cursor = _conn.cursor()
        query = """
            SELECT internal_id, name
            FROM vendor_metadata
            ORDER BY name;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        st.error(f"Failed to fetch vendor metadata names: {str(e)}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_vendor_filter_options(_conn) -> Dict[str, List[str]]:
    """
    Retrieve unique values for vendor metadata filters.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
    
    Returns:
        Dict containing lists of unique values for each filter:
        - sales_reps: List of unique sales representatives
        - states: List of unique billing states
        - dealer_terms: List of unique dealer terms
    """
    try:
        cursor = _conn.cursor()
        
        # Get unique sales representatives
        cursor.execute("""
            SELECT DISTINCT sales_rep
            FROM vendor_metadata
            WHERE sales_rep IS NOT NULL AND sales_rep != ''
            ORDER BY sales_rep;
        """)
        sales_reps = [row[0] for row in cursor.fetchall()]
        
        # Get unique states
        cursor.execute("""
            SELECT DISTINCT billing_state_province
            FROM vendor_metadata
            WHERE billing_state_province IS NOT NULL AND billing_state_province != ''
            ORDER BY billing_state_province;
        """)
        states = [row[0] for row in cursor.fetchall()]
        
        # Get unique dealer terms
        cursor.execute("""
            SELECT DISTINCT dealer_terms
            FROM vendor_metadata
            WHERE dealer_terms IS NOT NULL AND dealer_terms != ''
            ORDER BY dealer_terms;
        """)
        dealer_terms = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        
        return {
            'sales_reps': sales_reps,
            'states': states,
            'dealer_terms': dealer_terms
        }
    except Exception as e:
        st.error(f"Failed to retrieve filter options: {str(e)}")
        return {
            'sales_reps': [],
            'states': [],
            'dealer_terms': []
        }


def update_vendor_mapping(conn, vendor_name: str, internal_id: str) -> int:
    """
    Update all transactions with matching vendor name to set internal_id.
    Uses normalized vendor name matching.
    
    Args:
        conn: Database connection
        vendor_name: Vendor name to update (title case)
        internal_id: internal_id to assign
    
    Returns:
        Number of rows affected
    """
    try:
        cursor = conn.cursor()
        
        # Start transaction
        conn.autocommit = False
        
        query = """
            UPDATE transactions
            SET internal_id = %s
            WHERE INITCAP(LOWER(vendor)) = INITCAP(LOWER(%s))
              AND (internal_id IS NULL OR internal_id != %s);
        """
        cursor.execute(query, (internal_id, vendor_name, internal_id))
        affected_rows = cursor.rowcount
        
        # Commit transaction
        conn.commit()
        cursor.close()
        
        # Clear relevant caches
        get_vendor_internal_id.clear()
        
        return affected_rows
        
    except Exception as e:
        # Rollback on error
        conn.rollback()
        st.error(f"Failed to update vendor mapping: {str(e)}")
        raise
    finally:
        conn.autocommit = True


def undo_vendor_mapping(conn, vendor_name: str, internal_id: str) -> int:
    """
    Revert transactions back to NULL internal_id for specific vendor/internal_id pair.
    
    Args:
        conn: Database connection
        vendor_name: Vendor name to revert (title case)
        internal_id: internal_id to remove
    
    Returns:
        Number of rows affected
    """
    try:
        cursor = conn.cursor()
        
        # Start transaction
        conn.autocommit = False
        
        query = """
            UPDATE transactions
            SET internal_id = NULL
            WHERE INITCAP(LOWER(vendor)) = INITCAP(LOWER(%s))
              AND internal_id = %s;
        """
        cursor.execute(query, (vendor_name, internal_id))
        affected_rows = cursor.rowcount
        
        # Commit transaction
        conn.commit()
        cursor.close()
        
        # Clear relevant caches
        get_vendor_internal_id.clear()
        
        return affected_rows
        
    except Exception as e:
        # Rollback on error
        conn.rollback()
        st.error(f"Failed to undo vendor mapping: {str(e)}")
        raise
    finally:
        conn.autocommit = True


def get_fuzzy_match_candidates(
    vendor_name: str,
    all_metadata_names: List[Tuple[str, str]],
    top_n: int = 50
) -> List[Tuple[str, str, float]]:
    """
    Generate fuzzy match candidates using multiple matching strategies.
    
    Args:
        vendor_name: Vendor name to match
        all_metadata_names: List of (internal_id, name) tuples from vendor_metadata
        top_n: Number of top candidates to return
    
    Returns:
        List of (internal_id, name, score) tuples sorted by score descending
    """
    if not all_metadata_names or not vendor_name:
        return []
    
    # Extract just the names for fuzzy matching
    names_only = [name for _, name in all_metadata_names]
    name_to_id = {name: internal_id for internal_id, name in all_metadata_names}
    
    candidates_dict = {}  # {name: best_score}
    
    # Strategy 1: Standard ratio - good for overall similarity
    matches_ratio = process.extract(
        vendor_name,
        names_only,
        scorer=fuzz.ratio,
        limit=top_n
    )
    for match, score, _ in matches_ratio:
        candidates_dict[match] = max(candidates_dict.get(match, 0), score)
    
    # Strategy 2: Partial ratio - catches substrings (e.g., 'alberts' in 'alberts farmstead')
    matches_partial = process.extract(
        vendor_name,
        names_only,
        scorer=fuzz.partial_ratio,
        limit=top_n
    )
    for match, score, _ in matches_partial:
        # Boost score slightly for partial matches
        candidates_dict[match] = max(candidates_dict.get(match, 0), score * 1.1)
    
    # Strategy 3: Token set ratio - handles word order differences and subsets
    matches_token_set = process.extract(
        vendor_name,
        names_only,
        scorer=fuzz.token_set_ratio,
        limit=top_n
    )
    for match, score, _ in matches_token_set:
        candidates_dict[match] = max(candidates_dict.get(match, 0), score)
    
    # Strategy 4: Token sort ratio - handles reordering
    matches_token_sort = process.extract(
        vendor_name,
        names_only,
        scorer=fuzz.token_sort_ratio,
        limit=top_n
    )
    for match, score, _ in matches_token_sort:
        candidates_dict[match] = max(candidates_dict.get(match, 0), score)
    
    # Sort by best score and return top candidates with internal_id
    sorted_candidates = sorted(candidates_dict.items(), key=lambda x: x[1], reverse=True)
    return [(name_to_id[name], name, score) for name, score in sorted_candidates[:top_n]]


def format_currency(value):
    """Format a number as currency."""
    if pd.isna(value):
        return "$0.00"
    return f"${value:,.2f}"


def format_integer(value):
    """Format a number as integer."""
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"


def display_vendor_metadata(conn, metadata: Dict, internal_id: str):
    """
    Display vendor metadata information in a formatted container.
    Uses fallback to most recent transaction date if date_of_last_sale is not available.
    
    Args:
        conn: Database connection
        metadata: Dictionary containing vendor metadata fields
        internal_id: The internal_id for fallback transaction date lookup
    """
    st.markdown("### 📋 Vendor Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sales Representative:**")
        st.text(metadata.get('sales_rep') or 'Not Available')
        
        st.markdown("**Billing Address:**")
        address_lines = []
        if metadata.get('billing_address_1'):
            address_lines.append(metadata['billing_address_1'])
        if metadata.get('billing_address_2'):
            address_lines.append(metadata['billing_address_2'])
        
        city_state_zip = []
        if metadata.get('billing_city'):
            city_state_zip.append(metadata['billing_city'])
        if metadata.get('billing_state_province'):
            city_state_zip.append(metadata['billing_state_province'])
        if metadata.get('billing_zip'):
            city_state_zip.append(metadata['billing_zip'])
        
        if city_state_zip:
            address_lines.append(', '.join(city_state_zip))
        
        if address_lines:
            for line in address_lines:
                st.text(line)
        else:
            st.text('Not Available')
    
    with col2:
        st.markdown("**Date of Last Sale:**")
        date_of_last_sale = metadata.get('date_of_last_sale')
        
        if date_of_last_sale:
            # Use date from vendor_metadata
            st.text(date_of_last_sale.strftime('%Y-%m-%d'))
        else:
            # Fallback: use most recent transaction date
            most_recent_transaction = get_vendor_most_recent_transaction_date(conn, internal_id)
            if most_recent_transaction:
                st.text(f"{most_recent_transaction.strftime('%Y-%m-%d')} (from transactions)")
            else:
                st.text('Not Available')
        
        st.markdown("**Status:**")
        inactive = metadata.get('inactive', '').lower()
        if inactive in ['yes', 'y', 'true', '1']:
            st.markdown("🔴 **Inactive**")
        else:
            st.markdown("🟢 **Active**")
        
        st.markdown("**Dealer Terms:**")
        st.text(metadata.get('dealer_terms') or 'Not Available')
    
    st.markdown("---")


def search_vendor_metadata(searchterm: str, all_metadata_names: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Search function for streamlit-searchbox.
    Returns matching vendor names based on search term.
    
    Args:
        searchterm: User's search input
        all_metadata_names: List of (internal_id, name) tuples
    
    Returns:
        List of (internal_id, name) tuples matching the search term
    """
    if not searchterm:
        return []
    
    # Filter names that contain the search term (case-insensitive)
    matches = [
        (internal_id, name)
        for internal_id, name in all_metadata_names
        if searchterm.lower() in name.lower()
    ]
    
    # Return top 20 matches
    return matches[:20]


def display_vendor_mapping_interface(conn, vendor_name: str, all_metadata_names: List[Tuple[str, str]]):
    """
    Display manual vendor mapping interface with fuzzy matching and searchbox.
    
    Args:
        conn: Database connection
        vendor_name: The vendor name to map
        all_metadata_names: List of (internal_id, name) tuples from vendor_metadata
    """
    st.markdown("### ⚠️ Vendor Mapping Required")
    st.warning(
        f"The vendor **{vendor_name}** is not currently mapped to vendor metadata. "
        "Please select the correct vendor from the list below to establish the mapping."
    )
    
    # Get fuzzy match candidates
    with st.spinner("Finding similar vendors..."):
        candidates = get_fuzzy_match_candidates(vendor_name, all_metadata_names, top_n=50)
    
    if candidates:
        st.info(f"Found {len(candidates)} potential matches based on similarity.")
        
        # Create options for searchbox (show top candidates with scores)
        candidate_options = [
            (internal_id, f"{name} (similarity: {score:.0f}%)")
            for internal_id, name, score in candidates[:10]
        ]
        
        # Add searchbox for manual search
        st.markdown("**Search and select the correct vendor:**")
        st.caption("Type to search all vendors, or select from the suggested matches below.")
        
        # Create a selectbox with fuzzy candidates
        selected_option = st.selectbox(
            "Select vendor metadata to map:",
            options=[""] + [f"{internal_id}|{name}" for internal_id, name, _ in candidates],
            format_func=lambda x: x.split("|")[1] if "|" in x else "-- Select a vendor --",
            key=f"vendor_mapping_select_{vendor_name}"
        )
        
        if selected_option:
            selected_internal_id, selected_name = selected_option.split("|", 1)
            
            # Show preview of mapping
            st.info(f"**Mapping Preview:** '{vendor_name}' → '{selected_name}'")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Confirm Mapping", type="primary", key=f"confirm_map_{vendor_name}"):
                    try:
                        # Perform the mapping
                        affected_rows = update_vendor_mapping(conn, vendor_name, selected_internal_id)
                        
                        # Add to session state mapping history
                        if 'mapping_history' not in st.session_state:
                            st.session_state.mapping_history = []
                        
                        st.session_state.mapping_history.append({
                            'vendor_name': vendor_name,
                            'internal_id': selected_internal_id,
                            'metadata_name': selected_name,
                            'timestamp': datetime.now(),
                            'affected_rows': affected_rows
                        })
                        
                        st.success(f"✅ Successfully mapped {affected_rows} transaction(s) to '{selected_name}'!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to create mapping: {str(e)}")
    else:
        st.error("No vendor metadata candidates found. Please check the vendor_metadata table.")
    
    st.markdown("---")


def display_undo_panel(conn):
    """
    Display panel showing recent mappings with undo buttons.
    
    Args:
        conn: Database connection
    """
    if 'mapping_history' not in st.session_state or not st.session_state.mapping_history:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Recent Mappings")
    st.sidebar.caption("Mappings made in this session can be undone:")
    
    # Display mappings in reverse order (most recent first)
    for idx, mapping in enumerate(reversed(st.session_state.mapping_history)):
        with st.sidebar.expander(
            f"**{mapping['vendor_name']}** → {mapping['metadata_name'][:20]}...",
            expanded=False
        ):
            st.caption(f"Time: {mapping['timestamp'].strftime('%H:%M:%S')}")
            st.caption(f"Affected rows: {mapping['affected_rows']}")
            
            # Undo button
            if st.button(
                "🔙 Undo This Mapping",
                key=f"undo_{idx}_{mapping['vendor_name']}_{mapping['timestamp']}",
                help=f"Revert {mapping['affected_rows']} transaction(s) back to unmapped state"
            ):
                try:
                    # Perform undo
                    reverted_rows = undo_vendor_mapping(
                        conn,
                        mapping['vendor_name'],
                        mapping['internal_id']
                    )
                    
                    # Remove from history
                    actual_index = len(st.session_state.mapping_history) - 1 - idx
                    st.session_state.mapping_history.pop(actual_index)
                    
                    st.sidebar.success(f"✅ Undone! Reverted {reverted_rows} transaction(s).")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Failed to undo mapping: {str(e)}")


def main():
    """Main application logic."""
    
    # Initialize session state
    if 'mapping_history' not in st.session_state:
        st.session_state.mapping_history = []
    
    # Header
    st.title("📊 Vendor Inventory Analysis")
    st.markdown("Analyze inventory and sales patterns for individual vendors")
    st.markdown("---")
    
    # Initialize connection
    try:
        with st.spinner("Connecting to database..."):
            conn = get_database_connection()
        
        st.success("✅ Connected to database successfully!")
        
    except Exception:
        st.error("❌ Unable to connect to database. Please check your configuration.")
        st.stop()
    
    # Display undo panel in sidebar if there are any mappings
    display_undo_panel(conn)
    
    # Vendor filtering section
    st.subheader("Vendor Filters")
    
    # Get filter options
    with st.spinner("Loading filter options..."):
        filter_options = get_vendor_filter_options(conn)
    
    # Create filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        only_2025_vendors = st.checkbox(
            "Only show vendors with 2025 sales",
            value=True,
            help="When checked, only shows vendors who have transactions in 2025"
        )
    
    with col2:
        # Status filter
        selected_statuses = st.multiselect(
            "Status",
            options=["Active", "Inactive"],
            default=[],
            help="Filter vendors by active/inactive status"
        )
    
    # Additional filters in an expander for cleaner UI
    with st.expander("🔍 Advanced Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Sales Representative filter
            selected_sales_reps = st.multiselect(
                "Sales Representative",
                options=filter_options['sales_reps'],
                default=[],
                help="Filter vendors by sales representative"
            )
        
        with filter_col2:
            # State filter
            selected_states = st.multiselect(
                "State (Billing Address)",
                options=filter_options['states'],
                default=[],
                help="Filter vendors by billing state"
            )
        
        with filter_col3:
            # Dealer Terms filter
            selected_dealer_terms = st.multiselect(
                "Dealer Terms",
                options=filter_options['dealer_terms'],
                default=[],
                help="Filter vendors by dealer terms"
            )
    
    # Get list of vendors with filters applied
    with st.spinner("Loading vendor list..."):
        vendors = get_vendors(
            conn,
            only_2025=only_2025_vendors,
            sales_reps=selected_sales_reps if selected_sales_reps else None,
            states=selected_states if selected_states else None,
            dealer_terms=selected_dealer_terms if selected_dealer_terms else None,
            statuses=selected_statuses if selected_statuses else None
        )
    
    if not vendors:
        st.warning("No vendors found in the database.")
        st.stop()
    
    st.info(f"Found {len(vendors)} vendor(s) in the database")
    
    # Vendor selection dropdown with search
    st.subheader("Select a Vendor")
    selected_vendor = st.selectbox(
        "Choose a vendor to analyze:",
        options=vendors,
        index=0,
        help="Select a vendor from the dropdown to view their inventory analysis"
    )
    
    # Display inventory analysis for selected vendor
    if selected_vendor:
        st.markdown("---")
        
        # Check if vendor has internal_id mapping
        internal_id = get_vendor_internal_id(conn, selected_vendor)
        
        if internal_id:
            # Vendor is mapped - display metadata
            metadata = get_vendor_metadata_by_internal_id(conn, internal_id)
            if metadata:
                display_vendor_metadata(conn, metadata, internal_id)
            else:
                st.warning(f"Vendor is mapped to internal_id '{internal_id}' but metadata not found in vendor_metadata table.")
        else:
            # Vendor is not mapped - show mapping interface
            all_metadata_names = get_all_vendor_metadata_names(conn)
            if all_metadata_names:
                display_vendor_mapping_interface(conn, selected_vendor, all_metadata_names)
            else:
                st.error("No vendor metadata available in the database. Please populate the vendor_metadata table.")
        
        st.subheader(f"Inventory Analysis for: `{selected_vendor}`")
        
        with st.spinner(f"Analyzing inventory for {selected_vendor}..."):
            df = get_vendor_inventory_analysis(conn, selected_vendor)
        
        if df is not None and len(df) > 0:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SKUs Purchased in 2025", len(df))
            with col2:
                total_2025_qty = df["2025"].sum()
                st.metric("Total 2025 Quantity", f"{int(total_2025_qty):,}")
            with col3:
                total_opportunity = df["Opportunity"].sum()
                st.metric("Total Opportunity", f"${total_opportunity:,.2f}")
            with col4:
                total_oversupply = df["Total Oversupply"].sum()
                st.metric("Total Oversupply", f"{int(total_oversupply):,}")
            
            st.markdown("### Purchase History & Inventory Details")
            
            # Configure column display
            column_config = {
                "sku": st.column_config.TextColumn("SKU", width="small"),
                "Name": st.column_config.TextColumn("Name", width="large"),
                "2015": st.column_config.NumberColumn("2015", format="%d"),
                "2016": st.column_config.NumberColumn("2016", format="%d"),
                "2017": st.column_config.NumberColumn("2017", format="%d"),
                "2018": st.column_config.NumberColumn("2018", format="%d"),
                "2024": st.column_config.NumberColumn("2024", format="%d"),
                "2025": st.column_config.NumberColumn("2025", format="%d"),
                "On_Hand": st.column_config.NumberColumn("On Hand", format="%d"),
                "2025 Sales (all customers)": st.column_config.NumberColumn("2025 Sales (All)", format="%d"),
                "Base Price": st.column_config.NumberColumn("Base Price", format="$%.2f"),
                "Opportunity": st.column_config.NumberColumn("Opportunity", format="$%.2f"),
                "Total Oversupply": st.column_config.NumberColumn("Total Oversupply", format="%d")
            }
            
            # Sort dataframe by Opportunity (descending) to show highest opportunities first
            df_sorted = df.sort_values('Opportunity', ascending=False)
            
            # Display the dataframe with configuration
            st.dataframe(
                df_sorted,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
            # Download button (use sorted dataframe)
            csv = df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{selected_vendor}_inventory_analysis.csv",
                mime="text/csv",
                help="Download the inventory analysis data as a CSV file"
            )
            
        elif df is not None and len(df) == 0:
            st.warning(f"No SKUs found for {selected_vendor} in 2025.")
        else:
            st.error("Unable to load inventory analysis. Please try another vendor.")
    
    # Footer
    st.markdown("---")
    st.caption("💡 This analysis shows SKUs sold by the vendor in 2025, with historical sales data and opportunity calculations.")


if __name__ == "__main__":
    main()
