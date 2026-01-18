#!/usr/bin/env python3
"""
Create sample tables and fake data in Supabase for Notebook 04
Run this after successful connection to populate your database
Generates 500 rows per table with realistic fake data
"""

import os
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Get credentials
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# URL-encode credentials
encoded_user = quote_plus(POSTGRES_USER)
encoded_password = quote_plus(POSTGRES_PASSWORD)

# Build connection string with SSL
connection_string = (
    f"postgresql+psycopg2://{encoded_user}:{encoded_password}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode=require"
)

# Create engine
engine = create_engine(connection_string, pool_pre_ping=True)

# Data generation helpers
FIRST_NAMES = [
    'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
    'Ivy', 'Jack', 'Kate', 'Leo', 'Maya', 'Noah', 'Olivia', 'Paul',
    'Quinn', 'Ruby', 'Sam', 'Tara', 'Uma', 'Victor', 'Wendy', 'Xavier',
    'Yara', 'Zack', 'Ava', 'Ben', 'Chloe', 'David', 'Emma', 'Finn'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Thompson', 'White', 'Harris', 'Clark', 'Lewis', 'Walker', 'Hall'
]

CITIES = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
    'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
    'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
    'Seattle', 'Denver', 'Boston', 'Portland', 'Nashville', 'Miami', 'Atlanta'
]

PRODUCT_CATEGORIES = {
    'Electronics': [
        'Laptop', 'Desktop', 'Tablet', 'Smartphone', 'Monitor', 'Keyboard',
        'Mouse', 'Webcam', 'Headphones', 'Speaker', 'Microphone', 'Router',
        'External SSD', 'USB Hub', 'Graphics Card', 'RAM Module', 'Processor'
    ],
    'Furniture': [
        'Desk', 'Chair', 'Standing Desk', 'Bookshelf', 'Cabinet', 'Sofa',
        'Table', 'Lamp', 'Filing Cabinet', 'Monitor Stand', 'Footrest'
    ],
    'Accessories': [
        'Mouse Pad', 'USB Cable', 'HDMI Cable', 'Laptop Bag', 'Phone Case',
        'Screen Protector', 'Charging Cable', 'Cable Organizer', 'Desk Mat',
        'Laptop Stand', 'Phone Stand', 'Pen Holder', 'Notebook'
    ],
    'Office Supplies': [
        'Printer', 'Scanner', 'Shredder', 'Whiteboard', 'Markers', 'Pens',
        'Paper', 'Folders', 'Binders', 'Stapler', 'Calculator', 'Tape'
    ]
}

PRODUCT_ADJECTIVES = [
    'Pro', 'Premium', 'Ultra', 'Deluxe', 'Standard', 'Basic', 'Advanced',
    'Professional', 'Elite', 'Compact', 'Portable', 'Wireless', 'Smart'
]

def generate_customers(count=500):
    """Generate customer data."""
    customers = []
    for i in range(count):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        city = random.choice(CITIES)
        customers.append((first + ' ' + last, email, city))
    return customers

def generate_products(count=500):
    """Generate product data."""
    products = []
    for i in range(count):
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        base_name = random.choice(PRODUCT_CATEGORIES[category])

        # Add variation
        if random.random() > 0.5:
            adjective = random.choice(PRODUCT_ADJECTIVES)
            name = f"{base_name} {adjective}"
        else:
            name = base_name

        # Add model number or version
        if random.random() > 0.6:
            name += f" {random.choice(['2024', 'v2', 'v3', 'X', 'Plus', 'Max'])}"

        # Ensure unique names
        name = f"{name} #{i+1}"

        # Generate realistic prices based on category
        if category == 'Electronics':
            price = round(random.uniform(29.99, 2999.99), 2)
        elif category == 'Furniture':
            price = round(random.uniform(49.99, 1499.99), 2)
        elif category == 'Accessories':
            price = round(random.uniform(4.99, 99.99), 2)
        else:  # Office Supplies
            price = round(random.uniform(9.99, 499.99), 2)

        stock = random.randint(0, 500)
        products.append((name, category, price, stock))

    return products

def generate_orders(customer_count=500, product_count=500, order_count=500):
    """Generate order data."""
    orders = []
    start_date = datetime.now() - timedelta(days=365)  # Orders from last year

    for i in range(order_count):
        customer_id = random.randint(1, customer_count)
        product_id = random.randint(1, product_count)
        quantity = random.randint(1, 10)

        # Price will be calculated in SQL, but we'll use a placeholder
        # In real scenario, this would fetch the product price
        total_amount = round(random.uniform(10, 3000), 2)

        # Random date in the last year
        order_date = start_date + timedelta(days=random.randint(0, 365))

        orders.append((customer_id, product_id, quantity, total_amount, order_date))

    return orders

print("Generating 500 rows per table with realistic fake data...")
print("="*60)

try:
    with engine.connect() as conn:
        # Drop existing tables if they exist (for clean start)
        print("\n1. Cleaning up existing tables (if any)...")
        conn.execute(text("DROP TABLE IF EXISTS orders CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS products CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS customers CASCADE"))
        conn.commit()
        print("   ✓ Cleaned up")

        # Create customers table
        print("\n2. Creating customers table...")
        conn.execute(text("""
            CREATE TABLE customers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100),
                city VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
        print("   ✓ Created")

        # Create products table
        print("\n3. Creating products table...")
        conn.execute(text("""
            CREATE TABLE products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                category VARCHAR(50),
                price DECIMAL(10, 2),
                stock_quantity INTEGER
            )
        """))
        conn.commit()
        print("   ✓ Created")

        # Create orders table
        print("\n4. Creating orders table...")
        conn.execute(text("""
            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                product_id INTEGER REFERENCES products(id),
                quantity INTEGER,
                total_amount DECIMAL(10, 2),
                order_date TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
        print("   ✓ Created")

        # Generate and insert customers
        print("\n5. Generating customer data...")
        customers = generate_customers(500)
        print(f"   ✓ Generated {len(customers)} customers")

        print("   Inserting customers in batches...")
        batch_size = 100
        for i in range(0, len(customers), batch_size):
            batch = customers[i:i+batch_size]
            # Build SQL values (no quotes in generated data, so no escaping needed)
            values = ', '.join([
                f"('{name}', '{email}', '{city}')"
                for name, email, city in batch
            ])
            conn.execute(text(f"INSERT INTO customers (name, email, city) VALUES {values}"))
            print(f"   ✓ Inserted batch {i//batch_size + 1}/{(len(customers)-1)//batch_size + 1}")
        conn.commit()
        print(f"   ✓ Total customers inserted: {len(customers)}")

        # Generate and insert products
        print("\n6. Generating product data...")
        products = generate_products(500)
        print(f"   ✓ Generated {len(products)} products")

        print("   Inserting products in batches...")
        batch_size = 100
        for i in range(0, len(products), batch_size):
            batch = products[i:i+batch_size]
            # Build SQL values (no quotes in generated data, so no escaping needed)
            values = ', '.join([
                f"('{name}', '{category}', {price}, {stock})"
                for name, category, price, stock in batch
            ])
            conn.execute(text(f"INSERT INTO products (name, category, price, stock_quantity) VALUES {values}"))
            print(f"   ✓ Inserted batch {i//batch_size + 1}/{(len(products)-1)//batch_size + 1}")
        conn.commit()
        print(f"   ✓ Total products inserted: {len(products)}")

        # Generate and insert orders
        print("\n7. Generating order data...")
        orders = generate_orders(500, 500, 500)
        print(f"   ✓ Generated {len(orders)} orders")

        print("   Inserting orders in batches...")
        batch_size = 100
        for i in range(0, len(orders), batch_size):
            batch = orders[i:i+batch_size]
            values = ', '.join([
                f"({cid}, {pid}, {qty}, {amt}, '{date.strftime('%Y-%m-%d %H:%M:%S')}')"
                for cid, pid, qty, amt, date in batch
            ])
            conn.execute(text(f"INSERT INTO orders (customer_id, product_id, quantity, total_amount, order_date) VALUES {values}"))
            print(f"   ✓ Inserted batch {i//batch_size + 1}/{(len(orders)-1)//batch_size + 1}")
        conn.commit()
        print(f"   ✓ Total orders inserted: {len(orders)}")

        # Verify data
        print("\n8. Verifying data...")
        result = conn.execute(text("SELECT COUNT(*) FROM customers"))
        customer_count = result.scalar()
        print(f"   ✓ Customers: {customer_count}")

        result = conn.execute(text("SELECT COUNT(*) FROM products"))
        product_count = result.scalar()
        print(f"   ✓ Products: {product_count}")

        result = conn.execute(text("SELECT COUNT(*) FROM orders"))
        order_count = result.scalar()
        print(f"   ✓ Orders: {order_count}")

        print("\n" + "="*60)
        print("✅ SUCCESS! Sample database created with 500 rows per table!")
        print("="*60)
        print("\nDatabase Statistics:")
        print(f"  📊 Customers: {customer_count} rows")
        print(f"  📦 Products: {product_count} rows")
        print(f"  🛒 Orders: {order_count} rows")
        print(f"  💾 Total rows: {customer_count + product_count + order_count}")
        print("\nYou can now use these tables in Notebook 04 for text-to-SQL!")
        print("\nExample queries to try:")
        print('  - "How many customers are there?"')
        print('  - "What are the top 10 most expensive products?"')
        print('  - "How many orders were placed last month?"')
        print('  - "Which customer spent the most money?"')
        print('  - "What is the average price of Electronics products?"')
        print('  - "List all customers from New York who made orders"')
        print('  - "What products are low in stock (less than 50)?"')
        print('  - "Show total revenue by product category"')
        print('  - "Which products are most popular (most ordered)?"')
        print('  - "What is the order trend over time?"')

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're connected to Supabase")
    print("2. Check your credentials in .env file")
    print("3. Ensure you have CREATE TABLE permissions")
