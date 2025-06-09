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
        "mjere_predostrožnosti": "Izbjegavati masnu kozmetiku, redovno čistiti lice, koristiti blage proizvode."
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "Ime": "Aktinična keratoza, bazocelularni karcinom i druge zloćudne lezije",
        "uzrok": "Dugotrajno izlaganje UV zračenju.",
        "mjere_predostrožnosti": "Koristiti zaštitu od sunca (kreme, odjeća), izbjegavati sunce u podne."
    },
    "Atopic Dermatitis Photos": {
        "Ime": "Atopijski dermatitis",
        "uzrok": "Genetska predispozicija, suha koža i alergeni.",
        "mjere_predostrožnosti": "Redovno hidratizirati kožu, izbjegavati poznate alergene i nadražaje."
    },
    "Bullous Disease Photos": {
        "Ime": "Bulozne bolesti",
        "uzrok": "Autoimune reakcije koje uzrokuju stvaranje mjehura na koži.",
        "mjere_predostrožnosti": "Izbjegavati traumu kože, pratiti promjene i koristiti propisanu terapiju."
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "Ime": "Celulitis, impetigo i druge bakterijske infekcije",
        "uzrok": "Bakterije koje ulaze kroz oštećenu kožu.",
        "mjere_predostrožnosti": "Održavati higijenu, čistiti rane, izbjegavati kontakt sa zaraženim osobama."
    },
    "Eczema Photos": {
        "Ime": "Ekcem",
        "uzrok": "Genetski faktori i osjetljivost kože na iritanse.",
        "mjere_predostrožnosti": "Koristiti hidratantne kreme, izbjegavati iritanse i često tuširanje toplom vodom."
    },
    "Exanthems and Drug Eruptions": {
        "Ime": "Egzantemi i reakcije na lijekove",
        "uzrok": "Alergijske reakcije na lijekove ili virusne infekcije.",
        "mjere_predostrožnosti": "Izbjegavati poznate lijekove koji izazivaju reakciju, pratiti simptome i obratiti se ljekaru."
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "Ime": "Opadanje kose, alopecija i druge bolesti kose",
        "uzrok": "Autoimune bolesti, stres, hormonski disbalans.",
        "mjere_predostrožnosti": "Izbjegavati stres, hemijske tretmane i konzultovati dermatologa za terapiju."
    },
    "Herpes HPV and other STDs Photos": {
        "Ime": "Herpes, HPV i druge polno prenosive bolesti",
        "uzrok": "Virusne infekcije prenesene seksualnim kontaktom.",
        "mjere_predostrožnosti": "Koristiti zaštitu tokom odnosa, redovno se testirati i izbjegavati rizične kontakte."
    },
    "Light Diseases and Disorders of Pigmentation": {
        "Ime": "Bolesti izazvane svjetlošću i pigmentacijski poremećaji",
        "uzrok": "Poremećaji u proizvodnji melanina, UV zračenje ili genetski faktori.",
        "mjere_predostrožnosti": "Izbjegavati direktno sunce, koristiti zaštitu od UV zračenja i dermatološke savjete."
    },
    "Lupus and other Connective Tissue diseases": {
        "Ime": "Lupus i druge bolesti vezivnog tkiva",
        "uzrok": "Autoimune bolesti koje napadaju vezivno tkivo.",
        "mjere_predostrožnosti": "Izbjegavati sunčevu svjetlost, pratiti simptome i redovno uzimati propisanu terapiju."
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "Ime": "Melanom, rak kože, mladeži i pigmentne promjene",
        "uzrok": "Maligne promjene ćelija kože uzrokovane UV zračenjem i genetikom.",
        "mjere_predostrožnosti": "Koristiti zaštitne kreme, redovno pregledavati mladeže i posjećivati dermatologa."
    },
    "Nail Fungus and other Nail Disease": {
        "Ime": "Gljivice na noktima i druge bolesti noktiju",
        "uzrok": "Gljivične infekcije, povrede noktiju ili loša higijena.",
        "mjere_predostrožnosti": "Održavati higijenu noktiju, izbjegavati vlažna okruženja i koristiti antifungalne lijekove."
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "Ime": "Otrovni bršljan i drugi kontaktni dermatitis",
        "uzrok": "Kontakt sa iritantima poput otrovnog bršljana ili hemikalija.",
        "mjere_predostrožnosti": "Nositi zaštitnu odjeću, izbjegavati poznate iritanse i prati kožu nakon izlaganja."
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "Ime": "Psorijaza, lichen planus i srodne bolesti",
        "uzrok": "Autoimuni poremećaji koji ubrzavaju rast ćelija kože.",
        "mjere_predostrožnosti": "Izbjegavati stres, koristiti hidratantne kreme i slijediti terapiju propisanu od strane ljekara."
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "Ime": "Šuga, lajmska bolest i druge infestacije i ujedi",
        "uzrok": "Paraziti, ugrizi insekata ili zaraženi krpelji.",
        "mjere_predostrožnosti": "Održavati higijenu, izbjegavati kontakt sa zaraženima i koristiti zaštitu na otvorenom."
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "Ime": "Seboroične keratoze i drugi benigni tumori",
        "uzrok": "Benigne promjene na koži uzrokovane starenjem ili genetikom.",
        "mjere_predostrožnosti": "Pratiti promjene na koži i savjetovati se sa dermatologom kod sumnjivih promjena."
    },
    "Systemic Disease": {
        "Ime": "Sistemske bolesti",
        "uzrok": "Unutrašnje bolesti koje se manifestuju na koži (npr. dijabetes, bolesti jetre).",
        "mjere_predostrožnosti": "Kontrolisati osnovnu bolest, pratiti stanje kože i redovno se pregledati."
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "Ime": "Tinea, gljivice, kandidijaza i druge gljivične infekcije",
        "uzrok": "Gljivične infekcije kože ili sluznica.",
        "mjere_predostrožnosti": "Održavati suhu i čistu kožu, izbjegavati dijeljenje ličnih stvari i koristiti protugljivične lijekove."
    },
    "Urticaria Hives": {
        "Ime": "Urtikarija (koprivnjača)",
        "uzrok": "Alergijska reakcija na hranu, lijekove, stres ili druge podražaje.",
        "mjere_predostrožnosti": "Izbjegavati poznate alergene, koristiti antihistaminike i smanjiti stres."
    },
    "Vascular Tumors": {
        "Ime": "Vaskularni tumori",
        "uzrok": "Abnormalan rast krvnih sudova, često benignog karaktera.",
        "mjere_predostrožnosti": "Pratiti promjene, izbjegavati povrede i konsultovati se sa dermatologom za dalje korake."
    },
    "Vasculitis Photos": {
        "Ime": "Vaskulitis",
        "uzrok": "Upala krvnih sudova uzrokovana autoimunim ili infektivnim stanjima.",
        "mjere_predostrožnosti": "Redovno pratiti stanje, izbjegavati infekcije i slijediti preporuke ljekara."
    },
    "Warts Molluscum and other Viral Infections": {
        "Ime": "Bradavice, molluscum i druge virusne infekcije",
        "uzrok": "Virusne infekcije kože poput HPV-a ili Molluscum contagiosum.",
        "mjere_predostrožnosti": "Izbjegavati direktan kontakt sa zaraženima, održavati higijenu i jačati imunitet."
    }
}
