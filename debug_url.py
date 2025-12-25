url = 'https://segment.com'
url_lower = url.lower()
print(f'URL: {url_lower}')
print(f'Ends with /login: {url_lower.endswith("/login")}')
print(f'Ends with /login/: {url_lower.endswith("/login/")}')
print(f'Contains /login: {"/login" in url_lower}')

path_patterns = ['/login', '/cart', '/checkout', '/forgot-password']
for pattern in path_patterns:
    print(f'Pattern {pattern}: ends_with={url_lower.endswith(pattern)}, ends_with_slash={url_lower.endswith(pattern + "/")}')
