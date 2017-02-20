# SC2017 - Prepoznavanje karaktera Hiragane

Pokretanje projekta:

1. Pokrenuti gen_images.py - od velikih slika izdvaja sličice 150x150 koje sadrže jedan karakter.
    Te sličice se snimaju na disk na pogodan način.
2. Pokrenuti gen_test.py - od istih slika se izdvajaju sličice na kojima će se testirati algoritmi.
3. Pokrenuti prep_dataset.py - od malih sličica izdvojenih prvom skriptom, kreira dataset wrapper klasu
    i nju snima na disk u vidu .hdf5 fajla. Fajl se na dalje može koristiti za pristup datasetu.
4. Ručno izvršiti project.ipynb

Sadržaj fajlova u projektu:
