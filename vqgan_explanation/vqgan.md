# VQGAN
Grobe Idee: \
CNNs sind gut für lokale Strukturen im Bild, sind aber ort-invariant und bekommen keine gute grobe Struktur hin. \
Transformer können gute globale Verbindungen lernen (durch Attention-Mechanismus), der Rechenaufwand steigt aber quadratisch (nicht naiv für Bilder verwendbar)

&rarr; Beides intelligent kombinieren

# Bild als Codebook-Vektor darstellen: 
Intuition: Lokale Informationen müssen nicht Pixel-für-Pixel gespeichert sein, es reicht, wenn der Generator aus eine Codebook-Eintrag wieder "Fell" oder "Himmel" machen kann. Bilder können im Allgemeinen ~80% komprimiert werden und wir finden sie immernoch okay. 

Bild x, Encoder(=CNN) E, Quantisierung-Prozess q, Generator G\
Quantisierung hängt vom Codebook Z  ab\
$\hat{x} = G(q(E(x)))$\
Loss-Funktion: $L_{VQ}(E,G,Z) = ||x-\hat{x}||^2 + ||sg[E(x)] -z_q||_2^2 + ||sg[z_q]-E(x)||_2^2$\
* $||x-\hat{x}||^2$ Reconstruction Loss: Das generierte Bild sollte der Eingabe ähneln. Anstelle von Pixel-wise Loss hier Perceptual Loss: Berechnete Features aus DNNs sind gute Approximationen für menschliche Wahrnehmung und können verwendet werden, um Ähnlichkeit zwischen Bildern anzugeben
* $||sg[E(x)] -z_q||_2^2$ Stop Gradient Term
* $||sg[z_q]-E(x)||_2^2$ Commitment Loss: Bringt das Netz dazu, bei einer Repräsenation bleiben

Um gute Rekonstruktion sicherzustellen, wird ein GAN verwendet mit $L_{GAN}({E,G,Z},D) = log D(x) + log(1-D(\hat{x}))$

Gesamte Zielfunktion: \
![Latex-Formel für Objective Function](objective_Q.png "Gesamte Objective-Function für das Codieren von Bildern in Codebook-Einträge")\
wobei $\lambda$ adaptiv bestimmt, welches Ziel gerade besonders wichtig ist. 

# Bild-Generierung mit Transformer-Model:
Intuition: Erstellt gute Bilder mit "High-Level-Plan", weil der Attention-Mechanismus globale Verbindungen ermöglicht. 

Die Codebook-Features als Sequenz s von Indices $s_i$ darstellen, Vorhersage von Nachfolgern durch Transformer. \
Das Transformer-Model lernt quasi die Wahrscheinlichkeitsdichte, Inferenz sucht die Sequqenz mit der höchsten Wahrscheinlichkeit
$p(s) = \prod_i p(s_i|s_{<i})$, also Loss-Funktion $L_{\text{Transformer}}=$&#120124;$_{x~P(x)}[-log(p(s)]$


# Überblick
![Überblick über das Verfahren von VQGAN](https://compvis.github.io/taming-transformers/paper/teaser.png "Überblick")

# Fancy Ergebnisse
![](https://compvis.github.io/taming-transformers/images/article-Figure13-1.jpg)
## Fancy Features
1. Depth-to-image on RIN
2. Stochastic superresolution on IN
3. Semantic synthesis on S-FLCKR
4. Semantic synthesis on S-FLCKR
5. Edge-guided synthesis on IN.
![](https://compvis.github.io/taming-transformers/images/article-Figure6-1.jpg)
# Vorteile / Tricks im Paper
* Große Anzahl an Pixeln wird durch kompakte Codebook-Repräsentation abgefangen
* Sliding Window für *große* Bilder ermöglicht immernoch genug High-Level-Kontext
![](./sliding_attention.png)
* Einschränken der Wahrscheinlichkeitsdichte, über die maximiert wird für viele schöne Features wie
    * Class-Conditioned $p(s|c) = \prod_i p(s_i|s_{<i}, c)$
    * Image Completion, indem ein Teil der Sequenz s schon vorgegeben ist
* Separate Codebooks für einschränkende Init-Bilder lernen
    * Depth to image
    * Semantic Input Images 
    * Stochastic Superresolution
# Nachteile
* 

# Zahlen
* 85M bis 310M Parameter 
* |Codebook|=1024
* Länge der Sequenzen für Transformer = 16*16 = 256


## Tricks
### Kombination mit CLIP-Model
Problem: Generative Modelle können Bilder aus Rauschen generieren, aber wir wissen nicht, wie wir an die Bilder kommen, die wir haben wollen. 
CLIP: Contrastive Language–Image Pre-training
![](https://miro.medium.com/max/1400/1*IOOGa1YmHUo0P4ntmzmUjw.png)


# Quellen
* Paper "Taming Transformers for High-Resolution Image Synthesis" von Patrick Esser und Robin Rombach
* Artikel "The Illustrated VQGAN" von Lester James Miranda: https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/
* CLIP: https://openai.com/blog/clip/
VQGAN + CLIP Erklärung: https://alexasteinbruck.medium.com/explaining-the-code-of-the-popular-text-to-image-algorithm-vqgan-clip-a0c48697a7ff