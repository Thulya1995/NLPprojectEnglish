# -------------------------------------------------------
# Assignment (include number)
# Written by (include your name and student id)
# For COMP 472 Section (your lab section) – Summer 2020
# --------------------------------------------------------
from math import log
import pandas as pd
import nltk
import matplotlib.pyplot as plt

HackerNewsDataset = []
story_words = []
ask_hn_words = []
show_hn_words = []
poll_words = []
all_words = []
vocabulary = {}
sigma = 0.5
totalDistinctWords = []
post_in_2019 = {}
tokenizer = nltk.RegexpTokenizer(r'\w+')
numberOfLeftWord = []
accuracy = []


def csv_reader():
    #   ----------------------------------------------------------
    #   The csv_reader function read csv file from the file system.
    #   Tokenize each and every title in the data set.
    #   ----------------------------------------------------------

    global HackerNewsDataset
    global story_words
    global ask_hn_words
    global show_hn_words
    global poll_words
    global vocabulary
    global all_words
    global sigma
    global totalDistinctWords
    global post_in_2019
    global tokenizer
    global numberOfLeftWord
    global accuracy
    print("processing please wait.......")
    #   Read the csv file
    HackerNewsDataset = pd.read_csv("hackerNews.csv", encoding="ISO-8859-1")

    #   Iterate through the read dataset and tokenize titles and assign them into data list for further processing
    for i in range(0, HackerNewsDataset.__len__()):
        #   replace used to consider "ask hn" and "show hn" as single word
        title = HackerNewsDataset.iloc[i]['Title'].lower().replace("ask hn", "ask_hn").replace("show hn", "show_hn")
        tokens = tokenizer.tokenize(title)
        filteredTokens = []
        #   in order reduce the computation time I removed all the words with only numbers
        for word in tokens:
            if not word.isdigit():
                filteredTokens.append(word)
        if HackerNewsDataset.iloc[i]['year'] == 2018:
            if HackerNewsDataset.iloc[i]['Post Type'] == 'ask_hn':
                ask_hn_words.extend(filteredTokens)

            if HackerNewsDataset.iloc[i]['Post Type'] == 'show_hn':
                show_hn_words.extend(filteredTokens)

            if HackerNewsDataset.iloc[i]['Post Type'] == 'story':
                story_words.extend(filteredTokens)

            if HackerNewsDataset.iloc[i]['Post Type'] == 'poll':
                poll_words.extend(filteredTokens)
        else:
            #   generate the test data set
            post_in_2019[HackerNewsDataset.iloc[i]['Title']] = {
                'tokens': filteredTokens,
                'post_type': HackerNewsDataset.iloc[i]['Post Type']
            }
    #   add all words in four deferent classes into one single lisr for further works
    all_words.extend(show_hn_words)
    all_words.extend(ask_hn_words)
    all_words.extend(poll_words)
    all_words.extend(story_words)

    #   Call vocabulary_maker function to reduce the words gradually
    print("removing words with frequency 1..")
    vocabulary_maker(1)
    print("removing words with frequency less than or equal 5..")
    vocabulary_maker(5)
    print("removing words with frequency less than or equal 10..")
    vocabulary_maker(10)
    print("removing words with frequency less than or equal 15..")
    vocabulary_maker(15)
    print("removing words with frequency less than or equal 20..")
    vocabulary_maker(20)
    print(numberOfLeftWord)
    print(accuracy)
    #   setup the x axis and y axis data
    xs = [x for x in numberOfLeftWord]
    ys = [y for y in accuracy]
    # Set the x axis label of the current axis.
    plt.xlabel('Number of left words in corpus')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy of the classifier')
    # Set a title
    plt.title('Infrequent word filter')
    plt.plot(xs, ys)
    plt.show()

#   take a tag and based on the tag remove the words from the vocabulary
def vocabulary_maker(tag):

    all_words_copy = all_words.copy()
    story_words_copy = story_words.copy()
    ask_hn_words_copy = ask_hn_words.copy()
    show_hn_words_copy = show_hn_words.copy()
    poll_words_copy = poll_words.copy()
    for word in set(all_words):
        if tag == 1:
            if all_words.count(word) == 1:
                all_words_copy.remove(word)
                if word in story_words_copy:
                    story_words_copy.remove(word)
                if word in ask_hn_words_copy:
                    ask_hn_words_copy.remove(word)
                if word in show_hn_words_copy:
                    show_hn_words_copy.remove(word)
                if word in poll_words_copy:
                    poll_words_copy.remove(word)

        if tag == 5:
            if all_words.count(word) <= 5:
                all_words_copy.remove(word)
                if word in story_words_copy:
                    story_words_copy.remove(word)
                if word in ask_hn_words_copy:
                    ask_hn_words_copy.remove(word)
                if word in show_hn_words_copy:
                    show_hn_words_copy.remove(word)
                if word in poll_words_copy:
                    poll_words_copy.remove(word)
        if tag == 10:
            if all_words.count(word) <= 10:
                all_words_copy.remove(word)
                if word in story_words_copy:
                    story_words_copy.remove(word)
                if word in ask_hn_words_copy:
                    ask_hn_words_copy.remove(word)
                if word in show_hn_words_copy:
                    show_hn_words_copy.remove(word)
                if word in poll_words_copy:
                    poll_words_copy.remove(word)
        if tag == 15:
            if all_words.count(word) <= 15:
                all_words_copy.remove(word)
                if word in story_words_copy:
                    story_words_copy.remove(word)
                if word in ask_hn_words_copy:
                    ask_hn_words_copy.remove(word)
                if word in show_hn_words_copy:
                    show_hn_words_copy.remove(word)
                if word in poll_words_copy:
                    poll_words_copy.remove(word)
        if tag == 20:
            if all_words.count(word) <= 20:
                all_words_copy.remove(word)
                if word in story_words_copy:
                    story_words_copy.remove(word)
                if word in ask_hn_words_copy:
                    ask_hn_words_copy.remove(word)
                if word in show_hn_words_copy:
                    show_hn_words_copy.remove(word)
                if word in poll_words_copy:
                    poll_words_copy.remove(word)

    #   find the unique words in the all_words list
    unique = set(all_words_copy)
    vocabulary.clear()

    for word in sorted(unique):
        #   generate the probabilistic model with smoothed probability with sigma = 0.5
        vocabulary[word] = {

            'word': word,
            'freq_in_story': story_words_copy.count(word),
            'con_prob_in_story': ((story_words_copy.count(word) + sigma) / (
                    story_words_copy.__len__() + sigma * unique.__len__())),
            'fre_in_ask_hn': ask_hn_words_copy.count(word),
            'con_prob_in_ask_hn': ((ask_hn_words_copy.count(word) + sigma) / (
                    ask_hn_words_copy.__len__() + sigma * unique.__len__())),
            'freq_in_show_hn': show_hn_words_copy.count(word),
            'con_prob_in_show_hn': ((show_hn_words_copy.count(word) + sigma) / (
                    show_hn_words_copy.__len__() + sigma * unique.__len__())),
            'freq_in_poll': poll_words_copy.count(word),
            'con_prob_in_poll': ((poll_words_copy.count(word) + sigma) / (
                    poll_words_copy.__len__() + sigma * unique.__len__())),

        }
    numberOfLeftWord.append(all_words_copy.__len__())
    naive_bayes_classifier()

def naive_bayes_classifier():
    right_count = 0
    for title in post_in_2019:
        words = post_in_2019[title].get('tokens')
        original_post_type = post_in_2019[title].get('post_type')
        score_board = [0, 0, 0, 0]
        for word in words:
            if word in vocabulary:
                #   to calculate the score log is used to avoid the arithmetic underflow
                score_board[0] += log(vocabulary[word].get('con_prob_in_story'))
                score_board[1] += log(vocabulary[word].get('con_prob_in_ask_hn'))
                score_board[2] += log(vocabulary[word].get('con_prob_in_show_hn'))
                score_board[3] += log(vocabulary[word].get('con_prob_in_poll'))

        # find the estimated class for the given title
        if score_board.index(max(score_board)) == 0:
            post_type = 'story'
        elif score_board.index(max(score_board)) == 1:
            post_type = 'ask_hn'
        elif score_board.index(max(score_board)) == 2:
            post_type = 'show_hn'
        else:
            post_type = 'poll'
        if original_post_type == post_type:
            result = "right"
            right_count += 1
        else:
            result = "wrong"

    # accuracy is calculated by the following equation
    accuracy.append((right_count / post_in_2019.__len__() * 100).__round__(2))
    #   END OF THE PROGRAM


# Call the csv_rader method to start the execution
csv_reader()
