from classification_core import is_obvious_special_url

print('Testing segment.com:')
print(f'  is_obvious_special_url: {is_obvious_special_url("https://segment.com")}')
print(f'  is_obvious_special_url: {is_obvious_special_url("https://segment.com/")}')
