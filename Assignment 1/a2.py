import re

# Opening and Reading "test.txt"
text = open('data/test.txt').read()

# Replacing single newline with space
text = re.sub(r'\n', ' ', text)
# Replacing double newline with single
text = re.sub(r'  ', '\n', text)

start = 0
index = 0
content = list(text)
# Changing nested punctuation characters to different symbols
for i in range(len(content) - 1):
    # Condition for nested quotes
    if content[i] == "'":
        if re.match(r'[A-Za-z]', content[i + 1]) and re.match(r'[^\w]', content[i - 1]):
            start += 1
        elif re.match(r'[\s;.?\']', content[i + 1]):
            start -= 1
    # Replacing nested punctuation with new symbol
    elif content[i] == '.' and re.match(r'\d', content[i + 1]):
        content[i] = '#'
    elif content[i] == '.' and start > 0 and content[i + 1] != "'" and content[i + 1] != '\n':
        content[i] = '#'
    elif content[i] == '?' and start > 0 and content[i + 1] != "'":
        content[i] = '$'
    elif content[i] == '!' and start > 0 and content[i + 1] != "'":
        content[i] = '@'
    elif content[i] == '\n':
        start = 0

text = ''.join(content)
# Splitting based on pattern
text = re.split(r'\n|(?<![A-Z][A-Za-z]\.)(?<![A-Z]\.)(?<=\.)\s|(?<=\.\')\s|(?<=[?!]\')\n', text)
text = [sent for sent in text if sent and sent != ' ' and re.match(r'.*\w.*', sent)]
sent = ''
for i in range(len(text)):
    text[i] = text[i].replace('#', '.')
    text[i] = text[i].replace('$', '?')
    text[i] = '<s>' + text[i] + '</s>'
    sent += text[i] + '\n'

# Saving content to disk
file = open('output/output1(a)2.txt', 'w')
file.write(sent)
file.close()