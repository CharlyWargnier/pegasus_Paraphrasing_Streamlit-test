import streamlit as st

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

def load_pegasus_tokenizer():
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    return pegasus_tokenizer

def load_pegasus_model():
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    return pegasus_model

tok = load_pegasus_tokenizer()
model = load_pegasus_model()


# PEGASUS Summarization function
def pegasus_summarize(text):
    batch = tok.prepare_seq2seq_batch(src_texts = [text])
    # Hyperparameter Tuning
    gen = model.generate(
        **batch,max_length = 10, # max length of summary
        min_length = 5, # min length of summary
        do_sample = True,
        temperature = 3.0,
        top_k =30,
        top_p=0.70,
        repetition_penalty = 1.2,
        length_penalty = 5, # if more than 1 encourage model to generate #larger sequences
        num_return_sequences=1) # no of summary you want to generate
    # for forward pass: model(**batch)
    summary = tok.batch_decode(gen, skip_special_tokens=True)
    print(summary)

#text = 'Michael Jackson died of acute propofol and ... She advised the aide to send Jackson to a hospital. Arnold Klein said' #@param {type:"string"}
text = 'Michael Jackson died of acute a propofol ' #@param {type:"string"}

st.write('test')
st.write('test2')
st.write(text)
pegasus_summarize(text)
test = pegasus_summarize(text)
st.write(test)



