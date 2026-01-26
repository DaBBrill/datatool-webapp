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
from typing import List, Optional
from urllib.parse import urlparse


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
def get_vendors(_conn, only_2025: bool = True) -> List[str]:
    """
    Retrieve list of unique vendor names from transactions table.
    Normalizes vendor names to title case for consistent display.
    
    Args:
        _conn: Database connection (underscore prefix prevents hashing)
        only_2025: If True, only show vendors with 2025 sales (default: True)
    
    Returns:
        List[str]: Sorted list of unique normalized vendor names
    """
    try:
        cursor = _conn.cursor()
        
        # Conditionally add WHERE clause for 2025 filtering
        if only_2025:
            query = """
                SELECT DISTINCT INITCAP(LOWER(vendor)) AS normalized_vendor
                FROM transactions
                WHERE vendor IS NOT NULL
                  AND EXTRACT(YEAR FROM invoice_date) = 2025
                ORDER BY normalized_vendor;
            """
        else:
            query = """
                SELECT DISTINCT INITCAP(LOWER(vendor)) AS normalized_vendor
                FROM transactions
                WHERE vendor IS NOT NULL
                ORDER BY normalized_vendor;
            """
        
        cursor.execute(query)
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


def main():
    """Main application logic."""
    
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
    
    # Vendor filtering checkbox
    st.subheader("Vendor Filters")
    only_2025_vendors = st.checkbox(
        "Only show vendors with 2025 sales",
        value=True,
        help="When checked, only shows vendors who have transactions in 2025"
    )
    
    # Get list of vendors
    with st.spinner("Loading vendor list..."):
        vendors = get_vendors(conn, only_2025=only_2025_vendors)
    
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
