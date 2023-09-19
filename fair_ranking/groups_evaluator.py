#Class to evaluate the diversity in protected attributes
import yake

language = "en"
max_ngram_size = 1
num_keywords = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)

TEST_PATH = "D:/IntelliJ_Workspace/fairER/data/er_magellan/Structured/Walmart-Amazon/test.txt"

def extractValue(tuple, attribute_name):
    values_by_attribute = tuple.split(attribute_name + " VAL ")
    return values_by_attribute[1].split("COL")[0], values_by_attribute[2].split("COL")[0]

def getToken(text):
    language = "en"
    max_ngram_size = 1
    num_keywords = 1
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)

    return [kw[0] for kw in custom_kw_extractor.extract_keywords(text)]

with open(TEST_PATH, encoding="utf8") as file:
    candidates = [line.rstrip() for line in file]

left_values = set()
right_values = set()
for i in candidates:
    left, right = extractValue(i, "category")

    # print(getToken(left), getToken(right))

    if len(getToken(left)) > 0: left_values.add(getToken(left)[0])
    if len(getToken(right)) > 0: right_values.add(getToken(right)[0])
    # left_values.add(getToken(left))
    # right_values.add(getToken(right))

#Summary
print("left size: " + str(len(left_values)))
print("right size: " + str(len(right_values)))

print("---- Left ----")
for i in left_values:
    if i in right_values:
        print(i)

# print("---- Right ----")
# for i in right_values:
#     print(i)