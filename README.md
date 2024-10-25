**Cell Segmentation in Medical Images**


|Nr.|Autor(i)/An|Titlul articolului /proiectului|Aplicație /Domeniu|Tehnologii utilizate|Metodologie /Abordare|Rezultate|Limitări|Comentarii suplimentare|
|:---:|:-----:|:-------------------------------:|:------------------|:------------------:|:-------------------:|:-------:|:-----:|:----------------------:|
|1|Fatma Tuana Dogu,Hulya Dogan,Ramazan Ozgur Dogan,F. Sena Sezen/An aparitie 2024|Performance Improvement in Blood Cell
Segmentation with Deep Learning-based Image
Fusion Approach
|Segmentare imagini medicale|Deep Learning, Extended depth of focus(EDOF),CNN|Segmentarea cu deep learning , utilizand ,DOF cu hardware integrat pe microscop(suprapunerea tuturor imaginilor focusate in una singura), pe imagini optimizate din punct de vedere a focusului
-decodarea unor seturi de date cu diferite nivele de focus pentru a imbunatati imaginile de intrare
|S-a observat o imbunatatire a performantelor din punct de vedere a parametrilor , ACC,SEN,SPE atunci cand este utilizat deep learning , fata de metodele clasice de segmentare|Performanta variaza in functie de microscop|-|
|2|Navid Alemi Koohbanani,Mostafa Jahanifa,Neda Zamani Tajadin,
Nasir Rajpoot/An aparitie 2020
|NuClick: A Deep Learning Framework for
Interactive Segmentation of Microscopy Images
|Object segmentation|CNN(convolutional neuronal network),Inclusion Map.and Exclusion Map|Segmentarea celulelor sau a altor parti componente are corpului uman dorite pentru prelucrare  ulterioara utilizand specificarea manuala a unui punct care apartine acelui corp indiferent daca imaginea data ca intrare corespunde tiparului imaginilor pentru care a fost antrenata reteaua CNN sau nu.|NuClick a obtinut o performanta cel putin la fel de bună ca cea mai avansata tehnologie in domeniu.|O distanta de 50 px de la celula poate duce la o scadere semnificativa a performantei.
Eroarea nu poate fi masurata deoarece Nu Click se modifica constant
|util pentru reducerea timpului de selectie a datelor de intrare atunci cand dorim sa automatizam detectia de celule|
|3|Florian Roberta,Alexia Calovoulosc, Laurent Facqa,Fanny Decoeurd,Etienne Gontierd,Christophe,F. Grossetc,Baudouin Denis de Sennevillea|Enhancing Cell Instance Segmentation in Scanning Electron Microscopy Images via a Deep Contour Closing Operator|Cell segmentation from medical images|NnU-Net, A CNN Closing Operator(COp-Net,DIC-C2DH-HeLa cells dataset|Abordarea pentru segmentarea instanțelor celulare se bazează pe delimitarea limitelor celulei, care consta în doi pași rezumați. În primul rând, o hartă de probabilitate a conturului celulei în funcție de voxel este generată dintr-un SEM 3D imagine . Ulterior, se aplică un operator de închidere CNN (COp- Net) pentru a aborda golurile din celulă contururi. În cele din urmă, se aplică un algoritm de componentă conectată pentru a genera instanța celulei segmentare.|rezultatele au fost obţinute prin intermediul abordarii propuse şi a celei concurente folosind metode pe setul de date de validare privat nr2 ce au parametri de tip hiper impliciti. In intrarea SEM sunt raportate imaginea și segmentarea manuală, împreună cu rezultatele obținute folosind algoritmul Cellpose, decalajul concurent în pictarea metodei și modulul suplimentar propus COp-Net|Aceasta
performanță introdusa de Cop-Net poate fi îmbunătățită cu ajutorul arhitecturii 3D nnU-Net,dar folosirea ei introduce dificultati suplimentare legate de achiziția de date, costurile de memorie de calcul și timpul de procesare.
|-|
|4|Agnimitra Sen, Shyamali Mitra, Sukanta ChakrabortyDebashri Mondal, KC Santosh, Nibaran Das
/an aparitie 2022|Ensemble Framework for Unsupervised|segmentation in medical images|Algoritmi: K-means,K-means++, Mean Shift clustering. Fuzzy C-means clustering algorithm|Kmeans ofera clusterele nucleelo,Fuzzy C-means , decide clusterele finale dupa ce le suprapune , care vor deveni date de intrare pentru algortimul Kmeans++|Acest algoritm ofera rezultate promitatoare , depasind performantele algoritmilor nesupravegheati in acest domineiu cunoscuti pana in prezent |Nu sunt suficiente date medicale disponibile pentru public pentru antrenarea supravegheata , deci prin urmare eroarea este inca suficient de mare|-|
|5|Shuping Yuan, Vladimir Y.Mariano|Cell Segmentation Algorithm of Fully Connected U-Net ++ Network|cell segmentation|Deep learning,U-Net|Introduce conexiuni dense între nodurile rețelei, permițând ca informațiile din diferite etape să fie transmise eficient. Aceste conexiuni ajută la menținerea caracteristicilor semantice și detaliilor fine, reducând decalajul semantic între encoder și decoder. Optimizează viteza de testare prin păstrarea părților importante ale modelului și reducerea parametrilor, ceea ce permite selectarea între precizie și viteză în funcție de necesități. Utilizează o metodă de upsampling care îmbină caracteristicile de la diferite niveluri de rezoluție, ajutând la recuperarea detaliilor fine în timpul procesului de segmentare.|Fata de abordarile deja existente, numarul de date de intrare necesar pentru invatare scade odata la 10 incercari cu 20 %.|Este sensibil la date de intare de calitate slaba , performanța este afectata in cazul in care celulele se suprapun sau sunt neclare|-|





![proiect_pi_drawio(1)](https://github.com/user-attachments/assets/32fd36f0-f37e-4757-939b-071106beb7ec)

Descriere:
Proiectul va primi drept date de intrare imagini ce contin celule din diferite tesuturi.

In etapa de preprocesare imaginea de intrare va fi trecuta in grayscale pentru a se putea utiliza algoritmi mai simpli.
Deoarece datele de intare pot contine zgomote , se va face o filtrare a zgomotului pentru toate imaginile de intare.
Pentru imbunatatirea imaginii se va utiliza o egalizare de histograma. 
Pentru a pune in evidenta conturirile se va utiliza eventual binarizarea imaginii.

In etapa de procesarea se va realiza segmentarea de imagini vom incerca sa implementam un algoritm de tipul Watershed Segmentation: algoritm.

Optional: daca timpul ne va permite , vom cauta un ai antrenat deja pentru acest task pentru a compara rezultatele obtinute.
