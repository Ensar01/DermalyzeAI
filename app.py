import flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

class_names = [
    'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 
    'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 
    'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 
    'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 
    'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 
    'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 
    'Warts Molluscum and other Viral Infections'
]

class_details = {
    "Acne and Rosacea Photos": {
        "Ime": "Akne i rozacea",
        "uzrok": "Poremećaji lojnih žlijezda, hormonalne promjene ili bakterije.",
        "mjera_predostroznosti": "Izbjegavati masnu kozmetiku, redovno čistiti lice, koristiti blage proizvode."
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "Ime": "Aktinična keratoza, bazocelularni karcinom i druge zloćudne lezije",
        "uzrok": "Dugotrajno izlaganje UV zračenju.",
        "mjera_predostroznosti": "Koristiti zaštitu od sunca (kreme, odjeća), izbjegavati sunce u podne."
    },
    "Atopic Dermatitis Photos": {
        "Ime": "Atopijski dermatitis",
        "uzrok": "Genetska predispozicija, suha koža i alergeni.",
        "mjera_predostroznosti": "Redovno hidratizirati kožu, izbjegavati poznate alergene i nadražaje."
    },
    "Bullous Disease Photos": {
        "Ime": "Bulozne bolesti",
        "uzrok": "Autoimune reakcije koje uzrokuju stvaranje mjehura na koži.",
        "mjera_predostroznosti": "Izbjegavati traumu kože, pratiti promjene i koristiti propisanu terapiju."
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "Ime": "Celulitis, impetigo i druge bakterijske infekcije",
        "uzrok": "Bakterije koje ulaze kroz oštećenu kožu.",
        "mjera_predostroznosti": "Održavati higijenu, čistiti rane, izbjegavati kontakt sa zaraženim osobama."
    },
    "Eczema Photos": {
        "Ime": "Ekcem",
        "uzrok": "Genetski faktori i osjetljivost kože na iritanse.",
        "mjera_predostroznosti": "Koristiti hidratantne kreme, izbjegavati iritanse i često tuširanje toplom vodom."
    },
    "Exanthems and Drug Eruptions": {
        "Ime": "Egzantemi i reakcije na lijekove",
        "uzrok": "Alergijske reakcije na lijekove ili virusne infekcije.",
        "mjera_predostroznosti": "Izbjegavati poznate lijekove koji izazivaju reakciju, pratiti simptome i obratiti se ljekaru."
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "Ime": "Opadanje kose, alopecija i druge bolesti kose",
        "uzrok": "Autoimune bolesti, stres, hormonski disbalans.",
        "mjera_predostroznosti": "Izbjegavati stres, hemijske tretmane i konzultovati dermatologa za terapiju."
    },
    "Herpes HPV and other STDs Photos": {
        "Ime": "Herpes, HPV i druge polno prenosive bolesti",
        "uzrok": "Virusne infekcije prenesene seksualnim kontaktom.",
        "mjera_predostroznosti": "Koristiti zaštitu tokom odnosa, redovno se testirati i izbjegavati rizične kontakte."
    },
    "Light Diseases and Disorders of Pigmentation": {
        "Ime": "Bolesti izazvane svjetlošću i pigmentacijski poremećaji",
        "uzrok": "Poremećaji u proizvodnji melanina, UV zračenje ili genetski faktori.",
        "mjera_predostroznosti": "Izbjegavati direktno sunce, koristiti zaštitu od UV zračenja i dermatološke savjete."
    },
    "Lupus and other Connective Tissue diseases": {
        "Ime": "Lupus i druge bolesti vezivnog tkiva",
        "uzrok": "Autoimune bolesti koje napadaju vezivno tkivo.",
        "mjera_predostroznosti": "Izbjegavati sunčevu svjetlost, pratiti simptome i redovno uzimati propisanu terapiju."
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "Ime": "Melanom, rak kože, mladeži i pigmentne promjene",
        "uzrok": "Maligne promjene ćelija kože uzrokovane UV zračenjem i genetikom.",
        "mjera_predostroznosti": "Koristiti zaštitne kreme, redovno pregledavati mladeže i posjećivati dermatologa."
    },
    "Nail Fungus and other Nail Disease": {
        "Ime": "Gljivice na noktima i druge bolesti noktiju",
        "uzrok": "Gljivične infekcije, povrede noktiju ili loša higijena.",
        "mjera_predostroznosti": "Održavati higijenu noktiju, izbjegavati vlažna okruženja i koristiti antifungalne lijekove."
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "Ime": "Otrovni bršljan i drugi kontaktni dermatitis",
        "uzrok": "Kontakt sa iritantima poput otrovnog bršljana ili hemikalija.",
        "mjera_predostroznosti": "Nositi zaštitnu odjeću, izbjegavati poznate iritanse i prati kožu nakon izlaganja."
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "Ime": "Psorijaza, lichen planus i srodne bolesti",
        "uzrok": "Autoimuni poremećaji koji ubrzavaju rast ćelija kože.",
        "mjera_predostroznosti": "Izbjegavati stres, koristiti hidratantne kreme i slijediti terapiju propisanu od strane ljekara."
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "Ime": "Šuga, lajmska bolest i druge infestacije i ujedi",
        "uzrok": "Paraziti, ugrizi insekata ili zaraženi krpelji.",
        "mjera_predostroznosti": "Održavati higijenu, izbjegavati kontakt sa zaraženima i koristiti zaštitu na otvorenom."
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "Ime": "Seboroične keratoze i drugi benigni tumori",
        "uzrok": "Benigne promjene na koži uzrokovane starenjem ili genetikom.",
        "mjera_predostroznosti": "Pratiti promjene na koži i savjetovati se sa dermatologom kod sumnjivih promjena."
    },
    "Systemic Disease": {
        "Ime": "Sistemske bolesti",
        "uzrok": "Unutrašnje bolesti koje se manifestuju na koži (npr. dijabetes, bolesti jetre).",
        "mjera_predostroznosti": "Kontrolisati osnovnu bolest, pratiti stanje kože i redovno se pregledati."
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "Ime": "Tinea, gljivice, kandidijaza i druge gljivične infekcije",
        "uzrok": "Gljivične infekcije kože ili sluznica.",
        "mjera_predostroznosti": "Održavati suhu i čistu kožu, izbjegavati dijeljenje ličnih stvari i koristiti protugljivične lijekove."
    },
    "Urticaria Hives": {
        "Ime": "Urtikarija (koprivnjača)",
        "uzrok": "Alergijska reakcija na hranu, lijekove, stres ili druge podražaje.",
        "mjera_predostroznosti": "Izbjegavati poznate alergene, koristiti antihistaminike i smanjiti stres."
    },
    "Vascular Tumors": {
        "Ime": "Vaskularni tumori",
        "uzrok": "Abnormalan rast krvnih sudova, često benignog karaktera.",
        "mjera_predostroznosti": "Pratiti promjene, izbjegavati povrede i konsultovati se sa dermatologom za dalje korake."
    },
    "Vasculitis Photos": {
        "Ime": "Vaskulitis",
        "uzrok": "Upala krvnih sudova uzrokovana autoimunim ili infektivnim stanjima.",
        "mjera_predostroznosti": "Redovno pratiti stanje, izbjegavati infekcije i slijediti preporuke ljekara."
    },
    "Warts Molluscum and other Viral Infections": {
        "Ime": "Bradavice, molluscum i druge virusne infekcije",
        "uzrok": "Virusne infekcije kože poput HPV-a ili Molluscum contagiosum.",
        "mjera_predostroznosti": "Izbjegavati direktan kontakt sa zaraženima, održavati higijenu i jačati imunitet."
    }
}

image_processor=ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels = len(class_names)
)

#ucitavamo tezine koje smo prethodno sacuvali tokom treninga i nakon ovog koraka model je 100% identican onom koji je bio treniran 
model.load_state_dict(torch.load("ViTbestmodel_weights.pth", map_location=torch.device("cpu")))
model.eval()

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    #Nakon sto se slika pretvori u tenzor oduzmi srednju vrijednost (mean) i podijeli sa standardnom devijacijom, 
    #da bi slike bile u istom rasponu kao slike na kojima je ViT treniran
    #sto pomaze tacnosti
    #Npr. ako je image_processor.image_mean = [0.5, 0.5, 0.5]
    #image_processor.image_std = [0.5, 0.5, 0.5], onda:
    #piksel vrijednosti 0.6 na R kanalu postaje:
    #(0.6 - 0.5) / 0.5 = 0.2
    
    return transform(image).unsqueeze(0)

def classify_image(image_path):
    image = process_image(image_path)

    with torch.no_grad():
        outputs = model(image).logits
        probabilities = torch.nn.functional.softmax(outputs, dim = 1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_name = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100 

    ime = class_details.get(predicted_name, {}).get("uzrok", "Nepoznato")
    mjera_predostroznosti = class_details.get(predicted_name, {}).get("mjera_predostroznosti", "Nepoznato")
