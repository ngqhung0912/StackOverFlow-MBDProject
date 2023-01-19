"""
Last modified on 16th Jan.
Author: Hung.


"""
import re
import nltk
from nltk.tokenize import sent_tokenize



def word_count(content):
    return len(content.split())


def clean_data(content):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = content.lower()
    codeless_text = re.sub("<code>.*?</code>", "", text)  # remove code block
    cleantext = re.sub(rules, '', codeless_text)
    return cleantext


class Stats:
    def __init__(self, content):
        self.content = content


    def word_count(self):
        return len(self.content.split())

    def syllable_count(self):
        count = 0
        vowels = "aeiouy"
        word = self.content
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count

    def char_count(self):
        return len(self.content)

    def sentence_count(self):
        return len(sent_tokenize(self.content))


class Features:
    def __init__(self, content):
        self.text = content
        self.cleaned_text = clean_data(self.text)
        stats = Stats(self.cleaned_text)

        self.no_chars = stats.char_count()
        self.no_sentences = stats.sentence_count()
        self.no_words = stats.word_count()
        self.no_syllables = stats.syllable_count()

    def get_content(self):
        return self.cleaned_text

    # Coleman–Liau_index
    def coleman_liau(self):
        L = (self.no_chars / self.no_words) * 100
        S = (self.no_sentences / self.no_words) * 100
        return 0.0588 * L - 0.296 * S - 15.8

    # Flesch-Reading ease
    def flesch_ease(self):
        words_per_sent = self.no_words / self.no_sentences
        syll_per_word = self.no_syllables / self.no_words
        return 206.835 - (1.015 * words_per_sent) - (84.6 * syll_per_word)

    # Flesch-Reading ease
    def flesch_grade(self):
        words_per_sent = self.no_words / self.no_sentences
        syll_per_word = self.no_syllables / self.no_words
        return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59

    def code_percentage(content):
        # numbers between <code> tags / #no chars total
        pass



