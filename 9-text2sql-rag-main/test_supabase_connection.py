#!/usr/bin/env python3
"""
Quick test script to verify Supabase connection
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Get credentials
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")

print("Testing Supabase Connection")
print("="*50)
print(f"Host: {POSTGRES_HOST}")
print(f"Port: {POSTGRES_PORT}")
print(f"User: {POSTGRES_USER}")
print(f"Database: {POSTGRES_DB}")
print(f"Password: {'*' * len(POSTGRES_PASSWORD) if POSTGRES_PASSWORD else 'NOT SET'}")
print("="*50)

# Check if credentials are set
if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB]):
    print("\n❌ Missing credentials in .env file")
    print("Please set POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB")
    exit(1)

# URL-encode the password to handle special characters
encoded_password = quote_plus(POSTGRES_PASSWORD)
encoded_user = quote_plus(POSTGRES_USER)

# Build connection string with SSL and URL-encoded credentials
connection_string = (
    f"postgresql+psycopg2://{encoded_user}:{encoded_password}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode=require"
)

try:
    print("\n🔄 Attempting connection...")
    engine = create_engine(connection_string, pool_pre_ping=True)

    with engine.connect() as conn:
        # Test query
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]

        print("\n✅ Connection successful!")
        print(f"PostgreSQL Version: {version[:80]}...")

        # Get list of tables
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]

        if tables:
            print(f"\n📊 Found {len(tables)} table(s) in public schema:")
            for table in tables:
                print(f"  - {table}")
        else:
            print("\n📊 No tables found in public schema")
            print("You may need to create some tables first")

        print("\n✅ Supabase is ready for Notebook 04!")

except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your password is correct")
    print("2. Ensure no extra spaces in .env file")
    print("3. Verify your Supabase project is active")
    print("4. Try resetting your database password in Supabase settings")
    exit(1)
