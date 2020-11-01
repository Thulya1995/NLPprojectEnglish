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
    remove_word = open("remove_word.txt", "w")  # write mode

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
    remove_word.close()
    #   append all the words in four diferent array in to one array
    all_words.extend(show_hn_words)
    all_words.extend(ask_hn_words)
    all_words.extend(poll_words)
    all_words.extend(story_words)
    global frequency_array
    frequency_array = []

    #   add the words in all_words array into array called frequency_array
    for word in set(all_words):
        frequency_array.append((all_words.count(word), word))

    def takeSecond(elem):
        return elem[0]

    #   sort the frequency_array based on the word frequency
    frequency_array.sort(key=takeSecond, reverse=True)

    #   call vocabulary_maker function to reduce the number of most frequent words from the vocabulary
    print("removing to 5% most frequent word..")
    vocabulary_maker(5)
    print("removing to 10% most frequent word..")
    vocabulary_maker(10)
    print("removing to 15% most frequent word..")
    vocabulary_maker(15)
    print("removing to 20% most frequent word..")
    vocabulary_maker(20)
    print("removing to 25% most frequent word..")
    vocabulary_maker(25)
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
    plt.title('Most frequent word filter')
    plt.plot(xs, ys)
    plt.show()

#   take a tag and then remove all the words in corresponding precentage
def vocabulary_maker(tag):
    all_words_copy = all_words.copy()
    story_words_copy = story_words.copy()
    ask_hn_words_copy = ask_hn_words.copy()
    show_hn_words_copy = show_hn_words.copy()
    poll_words_copy = poll_words.copy()
    #   calculate the words to be removed from the vocabulary
    words_to_be_removed_5 = (5 * set(all_words).__len__()) / 100
    words_to_be_removed_10 = (10 * set(all_words).__len__()) / 100
    words_to_be_removed_15 = (15 * set(all_words).__len__()) / 100
    words_to_be_removed_20 = (20 * set(all_words).__len__()) / 100
    words_to_be_removed_25 = (25 * set(all_words).__len__()) / 100

    # if tag == 5 then remove top most 5% frequent words
    if tag == 5:
        for i in range (0, int(words_to_be_removed_5)):
            all_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in story_words_copy:
                story_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in ask_hn_words_copy:
                ask_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in show_hn_words_copy:
                show_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in poll_words_copy:
                poll_words_copy.remove(frequency_array[i][1])
    # if tag == 10 then remove top most 10% frequent words
    if tag == 10:
        for i in range(0, int(words_to_be_removed_10)):
            all_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in story_words_copy:
                story_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in ask_hn_words_copy:
                ask_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in show_hn_words_copy:
                show_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in poll_words_copy:
                poll_words_copy.remove(frequency_array[i][1])
    # if tag == 15 then remove top most 15% frequent words
    if tag == 15:
        for i in range(0, int(words_to_be_removed_15)):
            all_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in story_words_copy:
                story_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in ask_hn_words_copy:
                ask_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in show_hn_words_copy:
                show_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in poll_words_copy:
                poll_words_copy.remove(frequency_array[i][1])
    # if tag == 20 then remove top most 20% frequent words
    if tag == 20:
        for i in range(0, int(words_to_be_removed_20)):
            all_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in story_words_copy:
                story_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in ask_hn_words_copy:
                ask_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in show_hn_words_copy:
                show_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in poll_words_copy:
                poll_words_copy.remove(frequency_array[i][1])

    # if tag == 25 then remove top most 25% frequent words
    if tag == 25:
        for i in range(0, int(words_to_be_removed_25)):
            all_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in story_words_copy:
                story_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in ask_hn_words_copy:
                ask_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in show_hn_words_copy:
                show_hn_words_copy.remove(frequency_array[i][1])
            if frequency_array[i][1] in poll_words_copy:
                poll_words_copy.remove(frequency_array[i][1])

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
