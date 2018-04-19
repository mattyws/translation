f = open('questions-words.txt', 'r')
lines = []
for line in f:
    lines.append(line)
i = 0
topic_words = dict()
topic = ''
while i < len(lines):
    if ':' in lines[i]:
        topic = lines[i].strip()
        i += 1
        topic_words[topic] = set()
    else:
        splited_line = lines[i].split(' ')
        for word in splited_line:
            topic_words[topic].add(word.strip())
        i += 1

for key in topic_words.keys():
    print(key, topic_words[key])
    break
