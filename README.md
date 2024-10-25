#Cell Segmentation in Medical Images


 Outer pipes Cell padding

| Nr. | Autor(i)                                                                                                                                        | Titlul articolului/proiectului                                                                                   | Aplicatie/Domeniu                     | Tehnologii utilizate                                                | Metodologie/Abordare                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Rezultate                                                                                                                                                                                                                                                                                                                                                                                              | Limitari                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Comentarii suplimentare |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| 1   | Navid Alemi Koohbanani, Mostafa Jahanifa, Neda Zamani Tajadin4  and Nasir Rajpoot                                                               | NuClick: A Deep Learning Framework for Interactive Segmentation of Microscopy Images                             | Cell segmentation from medical images | NuClick,Grabcut,Graphcut

                                        | Folosesc o aplicatie numita NuClick ce utilizeaza un sistem de arhitectura de retea codificator-decodificator                                                                                                                                                                                                                                                                                                                                                                         | rezultatele sunt foarte robuste

împotriva distorsiunii aplicate în poziţia clicului. De exemplu Schimbarea

locația clicului cu 50 de pixeli face o scădere considerabilă

performanță care se poate datora aceluiași motiv ca și noi

a discutat cazul nucleelor, adică cantitatea de agitație este mai mare

decât raza medie a unor celule mici.                                                   | Din păcate, nu putem analiza cantitativ sensibilitatea

de NuClick la squiggle se schimbă, deoarece

modificările asociate nu sunt ușor de măsurat/parametrizable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                         |
| 2   | Florian Roberta, Alexia Calovoulosc, Laurent Facqa, Fanny Decoeurd, Etienne Gontierd,

Christophe F. Grossetc and Baudouin Denis de Sennevillea | Enhancing Cell Instance Segmentation in Scanning Electron

Microscopy Images via a Deep Contour Closing Operator | Cell segmentation from medical images | NnU-Net, A CNN Closing Operator(COp-Net,DIC-C2DH-HeLa cells dataset | Abordarea pentru segmentarea instanțelor celulare se bazează pe delimitarea limitelor celulei, care constă în

doi pași rezumați. În primul rând, o hartă de probabilitate a conturului celulei în funcție de voxel este generată dintr-un SEM 3D

imagine . Ulterior, se aplică un operator de închidere CNN (COp-Net) pentru a aborda golurile din celulă

contururi. În cele din urmă, se aplică un algoritm de componentă conectată pentru a genera instanța celulei

segmentare. | rezultatele au fost obţinute prin intermediul abordarii propuse şi a celei concurente folosind

metode pe setul de date de validare privat nr2 ce au parametri de tip hiper impliciti. In intrarea SEM

sunt raportate imaginea și segmentarea manuală, împreună cu rezultatele obținute folosind

algoritmul Cellpose, decalajul concurent în pictarea

metodei și modulul suplimentar propus COp-Net | Fiindca s-a folosit o arhitectură 2D pentru

operatorul de închidere a conturului celulei COp-Net s-au putut evita provocările în obținerea de date segmentate manual ce erau inainte și s-a putut facilita

comparația intre cele două tehnici menționate anterior.Aceasta performanță

 poate fi îmbunătățită cu ajutorul arhitecturii 3D nnU-Net,dar folosirea ei introduce dificultati suplimentare legate de achiziția de date, costurile de memorie de calcul și timpul de procesare. De asemenea, încercările noastre cu privire la 3D PDX pentru inferențe ortogonale și strategiile 2.5D nu au fost pe placul nostru din cauza interpolărilor necesare pentru

corectarea unor anizotropii semnificative a pixelilor în afara planului. |                         |
| 3   | Tuan Le Dinh,

Suk-Hwan Lee, Seong-Geun Kwon, Ki-Ryong Kwon                                                                                     | Cell Nuclei Segmentation in Cryonuseg dataset using Nested Unet with EfficientNet Encoder                        | Cell segmentation from medical images | Nested Unet model , the EfficientNet,Cryonuseg,GAN                  | Antrenăm modelul Nested Unet cu EfficientNet ca si

codificator prin alimentarea cu imagini de antrenament augmentate de la

Setul de date Cryonuseg. După am pregătit

model apoi am evaluat acuratețea cu testul

stabilit pe trei valori, valoarea Scorului de zaruri, valoarea AJI,

și metrica PQ cu evaluare de validare încrucișată de k ori.                                                                                                                                  | S-a creat un tabel in care s-au afisat rezultatul mediu la datul cu zarul,AJI si PQ pentru urmatoarele cazuri: Unet fără post-procesare,

 Unet cu postprocesare de bazin,

Distanţa Unet ,

Unet în două etape ,

Modelul nostru propus fără

post-procesare,

Modelul nostru propus cu

post-procesare a bazinului hidrografic                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                         |
| 4   | Fatma Tuana Dogu,Hulya Dogan,Ramazan Ozgur Dogan,F. Sena Sezen/An aparitie 2024                                                                 | Performance Improvement in Blood Cell

Segmentation with Deep Learning-based Image

Fusion Approach              | Segementare imagini medicale          | Deep Learning, Extended depth of focus(EDOF),CNN                    | Segmentarea cu deep learing , utilizand ,DOF cu hardware integrat pe microscop(suprapunerea tuturor imaginilor focusate in una singura), pe imagini optimizare din punct de vedere a focusului

\-incodarea unor seturi de date cu diferite nivele de focus pentru a imbunatati imaginile de intrare                                                                                                                                                                                  | S-a observat o imbunatatire a performantelor din punct de vedere a parametrilor , ACC,SEN,SPE atunci cand este utilizat deep learning , fata de metodele clasice de segmentare                                                                                                                                                                                                                         | Performanta variaza in functie de microscop                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | \-                      |
|     |                                                                                                                                                 |                                                                                                                  |                                       |                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                         |






![proiect_pi_drawio(1)](https://github.com/user-attachments/assets/32fd36f0-f37e-4757-939b-071106beb7ec)

Descriere:
Proiectul va primi drept date de intrare imagini ce contin celule din diferite tesuturi.

In etapa de preprocesare imaginea de intrare va fi trecuta in grayscale pentru a se putea utiliza algoritmi mai simpli.
Deoarece datele de intare pot contine zgomote , se va face o filtrare a zgomotului pentru toate imaginile de intare.
Pentru imbunatatirea imaginii se va utiliza o egalizare de histograma. 
Pentru a pune in evidenta conturirile se va utiliza eventual binarizarea imaginii.

In etapa de procesarea se va realiza segmentarea de imagini vom incerca sa implementam un algoritm de tipul Watershed Segmentation: algoritm.

Optional: daca timpul ne va permite , vom cauta un ai antrenat deja pentru acest task pentru a compara rezultatele obtinute.
