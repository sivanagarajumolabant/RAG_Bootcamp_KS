-- ================================================================
-- Supabase Setup Script for Text-to-SQL Project
-- ================================================================
-- Run this script in the Supabase SQL Editor BEFORE running
-- create_sample_tables.py to set up the database schema
-- ================================================================

-- Drop existing tables if they exist (for clean start)
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ================================================================
-- Create customers table
-- ================================================================
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    city VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add index on email for faster lookups
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_city ON customers(city);

-- ================================================================
-- Create products table
-- ================================================================
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock_quantity INTEGER DEFAULT 0 CHECK (stock_quantity >= 0)
);

-- Add index on category for faster filtering
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_price ON products(price);

-- ================================================================
-- Create orders table
-- ================================================================
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    total_amount DECIMAL(10, 2) NOT NULL CHECK (total_amount >= 0),
    order_date TIMESTAMP DEFAULT NOW()
);

-- Add indexes for better query performance
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_product_id ON orders(product_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);

-- ================================================================
-- Verification queries (optional - run these to verify setup)
-- ================================================================
-- Check tables were created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('customers', 'products', 'orders');

-- Check table structures
\d customers
\d products
\d orders

-- ================================================================
-- Notes:
-- ================================================================
-- 1. After running this script, run: python create_sample_tables.py
-- 2. The Python script will populate these tables with 500 rows each
-- 3. Total data: 500 customers + 500 products + 500 orders = 1,500 rows
-- 4. All foreign keys have CASCADE DELETE for easy cleanup
-- 5. Indexes added for better query performance with text-to-SQL
-- ================================================================
