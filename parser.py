import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import optparse
from bs4 import BeautifulSoup


nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')

# List all the stop words in english
stopwords = set(stopwords.words('english'))


# Preprocessing to perform pos_tagging, to be used in extract_valid_names   
def preprocess(document):
    # Remove stop words
    document = ' '.join([word for word in document.split() if word not in stopwords])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = [nltk.pos_tag(sentence) for sentence in sentences]
    return sentences


# It is used to get valid person names from pos_tagging
def extract_valid_names(document):
    global valid_names
    valid_names = []
    sentences = preprocess(document)
    for tagged_sentence in sentences:
        # Perform chunking to extract chunks of type PERSON
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    valid_names.append(' '.join([c[0] for c in chunk]))


# Regex to extract person name
def regex_extract_name(string):
    r = re.compile(r'(?:my\s)*\s*name\s*[:,]?\s*(?:is\s)*\s*([a-zA-Z\s]+)', re.IGNORECASE)
    name = r.findall(string)
    # If a match is found, return the first matched name
    if name and name[0] in valid_names:
        return name[0]
    return None


# Regex to extract phone number
def regex_extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{5}[-\s]??\d{5})')
    phone_numbers = r.findall(string)
    return phone_numbers


# Regex to extract email address
def regex_extract_email_addresses(string):
    r = re.compile(r'[\w\.-_]+@[\w]+\.[\w]+')
    return r.findall(string)


# Regex to extract property address
def regex_extract_property_address(string):
    # The pattern corresponds to US address format
    r = re.compile(r'[0-9]{1,5}(?:st|nd|rd|th)? [a-zA-Z\s]+, [a-zA-Z\s]+[,]? [A-Z]{2} [0-9]{4,5}')
    address = r.findall(string)
    # If a match is found, return the first matched address
    if address:
        return address[0]
    return None


# Regex to extract property details (number of beds)
def regex_extract_beds(string):
    r = re.compile(r'bed[s]?\s*[:]?\s*([0-9]+)', re.IGNORECASE)
    beds = r.findall(string)
    if beds:
        return beds[0]
    return None


# Regex to extract property details (number of baths)
def regex_extract_baths(string):
    r = re.compile(r'bath[s]?\s*[:]?\s*([0-9]+)', re.IGNORECASE)
    baths = r.findall(string)
    if baths:
        return baths[0]
    return None


# Function to get email addresses of buyer/seller
def get_email(sentences, index=0):
    for email_sentence in sentences[index:]:
        email = regex_extract_email_addresses(email_sentence)
        email_sentence = email_sentence.lower().split()
        # Checks to ensure the email is not the subscription email of the company
        if email and "add" not in  email_sentence and "subscribe" not in email_sentence and \
        "subscribed" not in email_sentence:
            return ", ".join(email)


# Function to get phone numbers of buyer/seller
def get_phone(sentences, index=0):
    for phone_sentence in sentences[index:]:
        phone = regex_extract_phone_numbers(phone_sentence)
        phone_sentence = phone_sentence.lower().split()
        # Checks to ensure the phone number isn't customer care number or doesn't belong to
        # agent (in case of buyer) 
        if phone and "agent" not in phone_sentence and "customer care" not in phone_sentence and \
        "call" not in phone_sentence:
            return", ".join(phone)


def main(file):
    
    # Used to store the text extracted
    document = ""

    # Used to store all sentences
    sentences = []

    # Used to store the output i.e, parsed details
    parsed_data = {}

    # Using Beautiful Soup to extract text from html file
    soup = BeautifulSoup(open('./' + file), 'html.parser')

    # Iterating through all tags to extract text from tags of type: font, div, p 
    for tag in soup.find_all():
        tag_name = tag.name
        if tag_name == "font" or tag_name == "div" or tag_name == "p":
            text = tag.text.replace("\n", "").strip()
            sentences.extend(text.split(". "))
            if text[-1] != ".":
                text += "."
            document = document + text + " "
    
    # Find all valid person names in the extracted text
    extract_valid_names(document)

    # Find the customer type (Buyer/Seller)
    if "seller" in document.lower() or "selling" in document.lower():
        parsed_data["type"] = "Seller"
    else:
        parsed_data["type"] = "Buyer"

    for index, name_sentence in enumerate(sentences):
        name = regex_extract_name(name_sentence)
        # Once a customer's name is found, the customer's details such as email and phone can be found.
        # Usually customer mentions his names before mentioning his email, phone and other details.  
        if name:
            parsed_data["name"] = name
            # Find customer's email addresses
            parsed_data["email"] = get_email(sentences, index)
            # Find customer's phone numbers
            parsed_data["phone"] = get_phone(sentences, index)
            # Break from the loop once a customer's details are found
            break
    
    # Sometimes, a customer might not have mentioned his name, but would have mentioned other details
    if "name" not in parsed_data:
        parsed_data["name"] = None
        parsed_data["email"] = get_email(sentences)
        parsed_data["phone"] = get_phone(sentences)
    
    # Used to ectract property address
    parsed_data["address"] = regex_extract_property_address(document)

    # Used to ectract property details (beds)
    parsed_data["beds"] = regex_extract_beds(document)

    # Used to ectract property details (baths)
    parsed_data["baths"] = regex_extract_baths(document)

    # Store the data in json format
    with open('./output.json', 'w+') as file:
        json.dump(parsed_data, file, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    # Initialize the parser to read the html file
    parser = optparse.OptionParser()
    # Accept the html file to be parsed as an argument
    parser.add_option(
        "-f", "--file",
        dest="file",
        help="html file location"
    )
    (options, args) = parser.parse_args()
    if (options.file is None):
        print("Please check if you have passed the html file")
    else:
        try:
            main(options.file)
        except Exception as e:
            print("Encountered following exception: ", e)
