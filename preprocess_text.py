# Purpose: Preprocess the text from the PDF file
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
# Open the PDF file
pdf_file = open('book.pdf', 'rb')

# Read the PDF file
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Extract the text from the PDF
#text = ''
#for page in range(pdf_reader.getNumPages()):
#    text += pdf_reader.getPage(page).extractText()

# Extract the text from the PDF with a counter per page
text = ''
for page in range(len(pdf_reader.pages)):
    text += pdf_reader.pages[page].extract_text()
    print('Page: ', page, ' of ', len(pdf_reader.pages))

# Remove the newlines from the text
text = text.replace('\n', '')

# Remove the single quotes from the text
text = text.replace("'", '')

# Remove the double quotes from the text
text = text.replace('"', '')

# save raw text to file
with open('raw_text.txt', 'w') as f:
    f.write(text)

# apply the nltk sentence tokenizer
sentences = nltk.sent_tokenize(text)

# Tokenize the text
tokens = word_tokenize(text)

# Summarize the text
print(len(tokens))
print(tokens[0:100])

# Close the PDF file
pdf_file.close()

# save the tokens to a file using pickle
import pickle
with open('tokens.pkl', 'wb') as f:
    pickle.dump(tokens, f)

# load the tokens from the file
with open('tokens.pkl', 'rb') as f:
    tokens = pickle.load(f)