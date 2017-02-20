# SC2017 - Prepoznavanje karaktera Hiragane

Pokretanje projekta:

1. Pokrenuti gen_images.py - od velikih slika izdvaja sličice 150x150 koje sadrže jedan karakter.
    Te sličice se snimaju na disk na pogodan način.
2. Pokrenuti gen_test.py - od istih slika se izdvajaju sličice na kojima će se testirati algoritmi.
3. Pokrenuti prep_dataset.py - od malih sličica izdvojenih prvom skriptom, kreira dataset wrapper klasu
    i nju snima na disk u vidu .hdf5 fajla. Fajl se na dalje može koristiti za pristup datasetu.
4. Ručno izvršiti project.ipynb

Sadržaj fajlova u projektu:

dataset_raw - Fascikla koja sadrži slike koje su prerađene u dataset
images50x50 - Fascikla koja sadrži slike za dataset, skalirane na dimenziju 50x50
models - Fascikla koja sadrži modele neuronskih mreža

CustomLogReg.py - Python fajl za implementaciju algoritma za logistic regression

Neural_Network_Zoo.ipynb - IPython Notebook u kome se nalaze modeli za sve implementacije neuronskih mreža

RNN.py - Python fajl u kojem se nalazi implementacija rekurentne neuronske mreže preko lasagne

Razvojna_Sveska.ipynb - IPython Notebook za testiranje neuronskih mreža

dataset.py - Python fajl koji formira dataset

directoryCreation.py - Python fajl koji formira fascikle za slike koje će biti korištene za dataset

gen_images.py - Python fajl za kropovanje sličica dimenzija 150x150 od kojih će se praviti dataset

gen_test.py - Python fajl koji generiše test primere sličica

hirautil.py - Python fajl u kom se nalazi implementacija erozije, ispunjavanje regiona i klasa Region

hnn.py - Python fajl preko kojeg se mogu pozivati neuronske mreže

poster.pdf - Poster projekta u pdf formatu

prep_dataset.py - Python fajl koji priprema sličice u memoriju pre kreiranja dataset fajla

project.ipynb - IPython notebook za razvoj obrade sličica

regions.ipynb - IPython Notebook koji sadrži primere kako radi regions i koristi se za izdvajanje regiona

regions.py - Python fajl koji sadrži sve funkcije za izdvajanje i spajanje regiona

seeding.txt - Pomoćna tekstualna datoteka koja sadrži mali primer random seedovanja koji se koristi kada se rezultati obučavanja posle žele replicirati



Od dodatnih biblioteka korištene su:
-Theano

-Keras

-Lasagne

-VGG16
