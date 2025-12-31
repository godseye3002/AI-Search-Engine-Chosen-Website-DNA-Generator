import os
from dotenv import load_dotenv
from supabase import create_client
import hashlib

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Check stored records
result = supabase.table('product_analysis_dna_google').select('id, product_id, input_data_hash, created_at').eq('product_id', '02f92e70-7b53-45b6-bdef-7ef36d8fc578').order('created_at', desc=True).execute()

print('=== DATABASE RECORDS ===')
for record in result.data:
    print(f'ID: {record["id"]}')
    print(f'Hash: {record.get("input_data_hash", "None")}')
    print(f'Created: {record["created_at"]}')
    print('---')

# Check current input rows
input_result = supabase.table('product_analysis_google').select('id').eq('product_id', '02f92e70-7b53-45b6-bdef-7ef36d8fc578').order('id').execute()

print('=== CURRENT INPUT ROWS ===')
ids = [str(row['id']) for row in input_result.data]
print(f'IDs: {ids}')
print(f'Sorted: {sorted(ids)}')
print(f'Combined: {",".join(sorted(ids))}')

current_hash = hashlib.sha256(",".join(sorted(ids)).encode('utf-8')).hexdigest()
print(f'Current Hash: {current_hash}')

# Check if any stored hash matches
for record in result.data:
    stored_hash = record.get('input_data_hash')
    if stored_hash == current_hash:
        print(f'✅ MATCH FOUND with record ID: {record["id"]}')
    else:
        print(f'❌ No match with record ID: {record["id"]} (stored: {stored_hash[:12] if stored_hash else "None"}... vs current: {current_hash[:12]}...)')
