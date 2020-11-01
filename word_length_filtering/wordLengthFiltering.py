from math import log
import pandas as pd
import nltk

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


def csv_reader():

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

    print("processing please wait.......")
    #   Read the csv file
    HackerNewsDataset = pd.read_csv("hackerNews.csv", encoding="ISO-8859-1")
    remove_word = open("wordlength_remove.txt", "w")  # write mode

    #   Iterate through the read dataset and tokenize titles and assign them into data list for further processing
    for i in range(0, HackerNewsDataset.__len__()):
        #   replace used to consider "ask hn" and "show hn" as single word
        title = HackerNewsDataset.iloc[i]['Title'].lower().replace("ask hn", "ask_hn").replace("show hn", "show_hn")
        tokens = tokenizer.tokenize(title)
        filteredTokens = []
        for word in tokens:
            if word.__len__() >= 9 or word.__len__() <= 2 or word.isdigit():
                remove_word.write(word + "\n")
            else:
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
    vocabulary_maker()


def vocabulary_maker():
    #   add each and every words in 4 different classes into one single list
    all_words.extend(show_hn_words)
    all_words.extend(ask_hn_words)
    all_words.extend(poll_words)
    all_words.extend(story_words)

    #   find the unique words in the all_words list
    unique = set(all_words)
    vocabularyText = open("wordlength_vocabulary.txt", "w")  # write mode
    for word in sorted(unique):
        #   write the vocabulary.text
        vocabularyText.write(word + "\n")
        #   generate the probabilistic model with smoothed probability with sigma = 0.5
        vocabulary[word] = {

            'word': word,
            'freq_in_story': story_words.count(word),
            'con_prob_in_story': ((story_words.count(word) + sigma) / (
                    story_words.__len__() + sigma * unique.__len__())),
            'fre_in_ask_hn': ask_hn_words.count(word),
            'con_prob_in_ask_hn': ((ask_hn_words.count(word) + sigma) / (
                    ask_hn_words.__len__() + sigma * unique.__len__())),
            'freq_in_show_hn': show_hn_words.count(word),
            'con_prob_in_show_hn': ((show_hn_words.count(word) + sigma) / (
                    show_hn_words.__len__() + sigma * unique.__len__())),
            'freq_in_poll': poll_words.count(word),
            'con_prob_in_poll': ((poll_words.count(word) + sigma) / (
                    poll_words.__len__() + sigma * unique.__len__())),

        }
    vocabularyText.close()

    write_model()


def write_model():
    #   the method write model write the generated vocabulary into a text file
    model_2018 = open("wordlength-model.txt", "w")  # write mode
    i = 1
    for word in vocabulary:
        line = vocabulary[word]
        model_2018.write(str(i) + "  " + word + "  " + str(line.get('freq_in_story')) + "  " + str(
            line.get('con_prob_in_story')) + "  " + str(line.get('fre_in_ask_hn')) + "  " + str(
            line.get('con_prob_in_ask_hn')) + "  " + str(line.get('freq_in_show_hn')) + "  " + str(
            line.get('con_prob_in_show_hn')) + "  " + str(line.get('freq_in_poll')) + "  " + str(
            line.get('con_prob_in_poll')) + "\n")
        i += 1
    model_2018.close()
    print("wordlength-model has been written with relevant data")
    naive_bayes_classifier()


def naive_bayes_classifier():
    baseline_result = open("wordlength-result.txt", "w", encoding="utf-8")  # write mode
    i = 1
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
        baseline_result.write(
            str(i) + "  " + title + "  " + post_type + "  " + str(score_board[0]) + "  " + str(
                score_board[1]) + "  " + str(score_board[2]) + "  " + str(
                score_board[3]) + "  " + original_post_type + "  " + result + "\n")
        i += 1
    baseline_result.close()
    # accuracy is calculated by the following equation
    print("Analysing complted with " + str((right_count / post_in_2019.__len__() * 100).__round__(2)) + "% accuracy.")
    print("Program Terminated Successfully")

    #   END OF THE PROGRAM


# Call the csv_rader method to start the execution
csv_reader()
