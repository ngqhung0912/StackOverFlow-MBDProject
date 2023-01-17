from codes.features import word_count, sentence_count, char_count


def test(text):
    print(word_count(text))
    print(char_count(text))
    print(sentence_count(text))


test("Bla, bla, bla.. We are here. Now? New sentence here! Okay")