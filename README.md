

```python

```

# TV Script Generation

In this project, a script is generated based on [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  A part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) was used to generate new scripts from the entire 9 seasons.

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( **.** )
- Comma ( **,** )
- Quotation Mark ( **"** )
- Semicolon ( **;** )
- Exclamation mark ( **!** )
- Question mark ( **?** )
- Left Parentheses ( **(** )
- Right Parentheses ( **)** )
- Dash ( **-** )
- Return ( **\n** )

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".

## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

You can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.
```
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```

### Batching
Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

>You can batch words using the DataLoader, but it will be up to you to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Your first `feature_tensor` should contain the values:
```
[1, 2, 3, 4]
```
And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
```
5
```
This should continue with the second `feature_tensor`, `target_tensor` being:
```
[2, 3, 4, 5]  # features
6             # target
```

## Generate TV Script
With the network trained and saved, you'll use it to generate a new, "fake" Seinfeld TV script in this section.

### Generate Text
To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. You'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!

### Generate a New Script
It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

You can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script. (You can also start with any other names you find in the original text file!)


```python
# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:43: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
    

    jerry: complete metal courtesy.
    
    jerry: i don't understand.
    
    elaine: what?
    
    elaine: well, it was nice meeting you.
    
    elaine: well, i'm sorry i'm late, i'm going down to the airport.
    
    jerry: well, it's a little freaked, i don't know what the hell's about.
    
    george: what are you talking about? what kind of a person does that have to be a lot nicer days..
    
    jerry:(pleading) oh, i can't believe i'm watching it. i'm gonna have to cancel the whole meeting, and i got a job interview.
    
    jerry: oh, i don't know least he'll be comfortable sexually.
    
    elaine: i thought you were going to have sex, you know, i just remembered that i wanted you to know how much duty is going?!
    
    jerry:(to elaine) hey, you wanna go upstairs to me, and i have a statue in the elevator.
    
    kramer: hey, you got any shredded coconut, and the best part, you want the saab.
    
    kramer: yeah.
    
    jerry: i don't have any insurance either.
    
    george: oh, come on, let's go.(to jerry) so you know, you better get together tomorrow afternoon, i was wondering if i had a pimple meeting you to drop dead.
    
    jerry: well you know, i don't want to get a bra.
    
    jerry: i thought you were jealous of a indistinct rye?
    
    elaine: yeah, it's just because i can't breathe to sleep.
    
    george: i don't think so. i can't sleep with you anymore. i don't think you want to get a glimpse of wine?
    
    george: yeah, it's a good idea.
    
    kramer:(quietly) i don't know what the hell do you think of a character?
    
    elaine: yeah.
    
    george: i don't understand.
    
    george: oh, i think it's possible.
    
    george: i don't understand.
    
    george: oh, no, no,
    


```python

```
