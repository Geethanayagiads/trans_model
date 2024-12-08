from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Initialize the Flask app
app = Flask(__name__)

# Set up the device and batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# Function to initialize the model and tokenizer
def initialize_model_and_tokenizer(ckpt_dir, quantization=None):
    if quantization == "4-bit":
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model

# Load the model and tokenizer for English to Indic translation
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir)

# Initialize Indic Processor
ip = IndicProcessor(inference=True)

# Batch translation function
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    return translations

# API endpoint for translation (Accepting raw text input)
@app.route('/translate', methods=['POST'])
def translate():
    # Get the raw text from POST request
    input_text = request.data.decode('utf-8')

    # Split input text into sentences (you can use a simple split or more complex sentence segmentation)
    input_sentences = input_text.split('\n')

    # Optional: Define source and target languages if not passed in the text
    src_lang = 'eng_Latn'
    tgt_lang = 'tam_Taml'

    if not input_sentences:
        return jsonify({'error': 'No sentences provided'}), 400

    # Perform the translation
    translations = batch_translate(input_sentences, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

    # Return the translated sentences as a plain text response
    return '\n'.join(translations)

# Root endpoint
@app.route('/')
def home():
    return {'message': 'Welcome to IndicTrans2 API!'}


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
