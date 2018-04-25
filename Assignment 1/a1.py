import re

# Opening and Reading File "test.txt"
file = open('data/test.txt', 'r')
content = file.read()

start=0
index = 0
content = list(content)
# Chaging inner quotes ' to #
for i in range(len(content)):
    if content[i] == "'":
        # Opening quote ' condition
        if i+1 <= len(content)-1 and re.match(r'[A-Za-z]', content[i+1]) and re.match(r'[^\w]', content[i-1]):
            start+=1
            if start != 1:
                index = i
        # Closing quote ' condition
        elif i+1 <= len(content)-1 and re.match(r'[\s;.?\']', content[i+1]):
            if start > 0:
                start-=1
                if start > 0:
                    content[index] = '#'
                    content[i] = '#'
            # apostrophe at end of the word
            else:
                content[i]='#'
    # Don't propagate error in next sentence
    elif content[i] == '\n' and i+1 <= len(content)-1 and content[i+1] == '\n':
        start = 0


content = ''.join(content)
# Replacing single quote ' to double quotes "
pat = re.compile(r"(?<=\s)(?<!\w\s)'(.+?)'(?=\s)", flags=re.DOTALL)
content = pat.sub(r'"\1"', content)
# Replacing old # to '
content = re.sub(r"#", "'", content)
# Saving content to disk
file = open('output/output1(a)1.txt', 'w')
file.write(content)
file.close()