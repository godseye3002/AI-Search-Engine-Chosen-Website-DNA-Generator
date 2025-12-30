import os
from dotenv import load_dotenv
from supabase import create_client
import hashlib

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Check current input rows
input_result = supabase.table('product_analysis_google').select('id').eq('product_id', '02f92e70-7b53-45b6-bdef-7ef36d8fc578').order('id').execute()

print('=== CURRENT INPUT ROWS ===')
ids = [str(row['id']) for row in input_result.data]
print(f'IDs: {ids}')
print(f'Sorted: {sorted(ids)}')
print(f'Combined: {",".join(sorted(ids))}')

current_hash = hashlib.sha256(",".join(sorted(ids)).encode('utf-8')).hexdigest()
print(f'Current Hash: {current_hash}')

# Check all stored records
result = supabase.table('product_analysis_dna_google').select('id, input_data_hash, created_at').eq('product_id', '02f92e70-7b53-45b6-bdef-7ef36d8fc578').order('created_at', desc=True).execute()

print('\n=== STORED RECORDS ===')
for record in result.data:
    stored_hash = record.get('input_data_hash')
    match = "✅ MATCH" if stored_hash == current_hash else "❌ NO MATCH"
    print(f'ID: {record["id"][:8]}... | Hash: {stored_hash[:12] if stored_hash else "None"}... | {match} | Created: {record["created_at"]}')
