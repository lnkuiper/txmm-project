def extract_features(text):
    bag_of_words = [x for x in wordpunct_tokenize(text)]
    sentences = [x for x in sent_tokenize(text)]

    features = []
    emoticons = [':-)',':)','=)',':-d',':d','=d','x)','xd',':-(',':(','=(',':\'-(',':\'(',';-)',
                ';)',':-p',':p','=p',';-p',';p','^^','>:-(','>:(','>:c',':-|',':|','-.-',':-s',
                ':s','=s','x.x',':-o',':o',':-o',':o','o.o','o_o','-.-','-,-','-.-"','-,-"','-_-']

    # Feature 22: total emoticon count
    features.append(sum([any(c in w.lower() for c in emoticons) for w in bag_of_words]))

    # Feature 23: Emoticon count
    features += [text.lower().count(e) for e in emoticons]

    # Feature 20: count of each specific digit
    features += [text.count(x) for x in [str(y) for y in range(10)]]

    # Feature 4: average words per sentence
    features.append(len(bag_of_words)/len(sentences))

    # Features 5-9: count the number of digits, spaces, lowercase, uppercase and other characters
    features.append(sum(c.isdigit() for c in text))
    features.append(sum(c.isspace() for c in text))
    features.append(sum(c.islower() for c in text))
    features.append(sum(c.isupper() for c in text))
    features.append(len(text) - sum(features[4:8])) # other

    # Features 10: counts of each upper- and lowercase letter
    features += [text.count(l) for l in list(string.ascii_lowercase) + list(string.ascii_uppercase)]

    # Feature 11: amount of words without vowels
    features.append(sum([any(c in w.lower() for c in r'aeiou') for w in bag_of_words]))

    # Features 12: punctuation count
    features += [text.count(p) for p in ['.',',','?','!','(',')',':',';','\'','\"']]

    # Feature 13: all uppercase word count
    features.append(sum(w.isupper() for w in bag_of_words))

    # Features 16-17: 'he' and 'she' frequency
    features.append(sum(w == 'he' for w in bag_of_words))
    features.append(sum(w == 'she' for w in bag_of_words))

    # Feature 18: amount of unique words
    features.append(len(set(bag_of_words)))

    # Feature 19: count of each stopword
    features += [text.count(sw) for sw in stop_words]

    # Feature 21: POS tag count
    all_tags = ['CC','CD','DT','EX','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNS','PDT','POS','PRP',
                'RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WRB']
    tags = [p[1] for p in pos_tag(bag_of_words)]
    features += [tags.count(pos) for pos in all_tags]

    return features