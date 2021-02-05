# 0x0B. Face Verification


## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/shyLM_J-1zTxj3VZCpkb1g "explain to anyone"), **without the help of Google**:

### General

*   What is face recognition?
*   What is face verification?
*   What is `dlib`?
*   How to use `dlib` for detecting facial landmarks
*   What is an affine transformation?
*   What is face alignment?
*   Why would you use facial alignment?
*   How to use `opencv-python` for face alignment
*   What is one-shot learning?
*   What is triplet loss?
*   How to create custom Keras layers
*   How to add losses to a custom Keras layer
*   How to create a Face Verification system
*   What are the ethical quandaries regarding facial recognition/verification?

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15), `tensorflow` (version 1.12), `opencv-python` (version 4.1.0.25), and `dlib` (version 19.17.0)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   All your files must be executable
*   The length of your files will be tested using `wc`

## Install dlib 19.17.0

    sudo apt-get update
    sudo apt-get install -y build-essential cmake
    sudo apt-get install -y libopenblas-dev liblapack-dev 
    sudo apt-get install -y libx11-dev libgtk-3-dev
    sudo apt-get install -y python3 python3-dev python3-pip
    pip install --user dlib==19.17.0






* * *

## Tasks



#### 0\. Load Images <span class="alert alert-warning mandatory-optional">mandatory</span>

Write the function `def load_images(images_path, as_array=True):` that loads images from a directory or file:

*   `images_path` is the path to a directory from which to load images
*   `as_array` is a boolean indicating whether the images should be loaded as one `numpy.ndarray`
    *   If `True`, the images should be loaded as a `numpy.ndarray` of shape `(m, h, w, c)` where:
        *   `m` is the number of images
        *   `h`, `w`, and `c` are the height, width, and number of channels of all images, respectively
    *   If `False`, the images should be loaded as a `list` of individual `numpy.ndarray`s
*   All images should be loaded in RGB format
*   The images should be loaded in alphabetical order by filename
*   Returns: `images`, `filenames`
    *   `images` is either a `list`/`numpy.ndarray` of all images
    *   `filenames` is a `list` of the filenames associated with each image in `images`
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 0-main.py
    #!/usr/bin/env python3

    from utils import load_images
    import matplotlib.pyplot as plt

    images, filenames = load_images('HBTN', as_array=False)
    print(type(images), len(images))
    print(type(filenames), len(filenames))
    idx = filenames.index('KirenSrinivasan.jpg')
    print(idx)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i+idx])
        plt.title(filenames[i+idx])
    plt.tight_layout()
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./0-main.py
    <class 'list'> 385
    <class 'list'> 385
    195
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `utils.py`






#### 1\. Load CSV <span class="alert alert-warning mandatory-optional">mandatory</span>

Also in `utils.py`, write a function `def load_csv(csv_path, params={}):` that loads the contents of a `csv` file as a list of lists:

*   `csv_path` is the path to the `csv` to load
*   `params` are the parameters to load the `csv` with
*   Returns: a list of lists representing the contents found in `csv_path`
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 1-main.py
    #!/usr/bin/env python3

    from utils import load_csv

    triplets = load_csv('FVTriplets.csv')
    print(type(triplets), len(triplets))
    print(triplets[:10])
    alexa@ubuntu-xenial:0x0B-face_verification$ ./1-main.py
    <class 'list'> 5306
    [['AndrewMaring', 'AndrewMaring0', 'ArthurDamm0'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm1'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm2'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm3'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm4'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm5'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm6'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm0'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm1'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm2']]
    alexa@ubuntu-xenial:0x0B-face_verification$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `utils.py`






#### 2\. Initialize Face Align <span class="alert alert-warning mandatory-optional">mandatory</span>

Create the class `FaceAlign`:

*   class constructor `def __init__(self, shape_predictor_path):`
    *   `shape_predictor_path` is the path to the `dlib` shape predictor model
    *   Sets the public instance attributes:
        *   `detector` - contains `dlib`‘s default face detector
        *   `shape_predictor` - contains the `dlib.shape_predictor`
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 2-main.py
    #!/usr/bin/env python3

    from align import FaceAlign

    fa = FaceAlign('models/landmarks.dat')
    print(type(fa.detector))
    print(type(fa.shape_predictor))
    alexa@ubuntu-xenial:0x0B-face_verification$ ./2-main.py
    <class 'dlib.fhog_object_detector'>
    <class 'dlib.shape_predictor'>
    alexa@ubuntu-xenial:0x0B-face_verification$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `align.py`






#### 3\. Detect Faces <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `FaceAlign`:

*   public instance method `def detect(self, image):` that detects a face in an image:
    *   `image` is a `numpy.ndarray` of rank 3 containing an image from which to detect a face
    *   Returns: a `dlib.rectangle` containing the boundary box for the face in the image, or `None` on failure
        *   If multiple faces are detected, return the `dlib.rectangle` with the largest area
        *   If no faces are detected, return a `dlib.rectangle` that is the same as the image
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 3-main.py
    #!/usr/bin/env python3

    from align import FaceAlign
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fa = FaceAlign('models/landmarks.dat')
    test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
    box = fa.detect(test_img)
    print(type(box))
    plt.imshow(test_img)
    ax = plt.gca()
    rect = Rectangle((box.left(), box.top()), box.width(), box.height(), fill=False)
    ax.add_patch(rect)
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./3-main.py
    <class 'dlib.rectangle'>
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `align.py`






#### 4\. Find Landmarks <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `FaceAlign`:

*   public instance method `def find_landmarks(self, image, detection):` that finds facial landmarks:
    *   `image` is a `numpy.ndarray` of an image from which to find facial landmarks
    *   `detection` is a `dlib.rectangle` containing the boundary box of the face in the image
    *   Returns: a `numpy.ndarray` of shape `(p, 2)`containing the landmark points, or `None` on failure
        *   `p` is the number of landmark points
        *   `2` is the x and y coordinates of the point
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 4-main.py
    #!/usr/bin/env python3

    from align import FaceAlign
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fa = FaceAlign('models/landmarks.dat')
    test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
    box = fa.detect(test_img)
    landmarks = fa.find_landmarks(test_img, box)
    print(type(landmarks), landmarks.shape)
    plt.imshow(test_img)
    ax = plt.gca()
    for landmark in landmarks:
        ax.add_patch(Circle(landmark))
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./4-main.py
    <class 'numpy.ndarray'> (68, 2)
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `align.py`






#### 5\. Align Faces <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `FaceAlign`:

*   public instance method `def align(self, image, landmark_indices, anchor_points, size=96):` that aligns an image for face verification:
    *   `image` is a `numpy.ndarray` of rank 3 containing the image to be aligned
    *   `landmark_indices` is a `numpy.ndarray` of shape `(3,)` containing the indices of the three landmark points that should be used for the affine transformation
    *   `anchor_points` is a `numpy.ndarray` of shape `(3, 2)` containing the destination points for the affine transformation, scaled to the range `[0, 1]`
    *   `size` is the desired size of the aligned image
    *   Returns: a `numpy.ndarray` of shape `(size, size, 3)` containing the aligned image, or `None` if no face is detected
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 5-main.py
    #!/usr/bin/env python3

    from align import FaceAlign
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np

    fa = FaceAlign('models/landmarks.dat')
    test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
    anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
    aligned = fa.align(test_img, np.array([36, 45, 33]), anchors, 96)
    plt.imshow(aligned)
    ax = plt.gca()
    for anchor in anchors:
        ax.add_patch(Circle(anchor * 96, 1))
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./5-main.py
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `align.py`






#### 6\. Save Files <span class="alert alert-warning mandatory-optional">mandatory</span>

Also in `utils.py`, write a function `def save_images(path, images, filenames):` that saves images to a specific path:

*   `path` is the path to the directory in which the images should be saved
*   `images` is a `list`/`numpy.ndarray` of images to save
*   `filenames` is a `list` of filenames of the images to save
*   Returns: `True` on success and `False` on failure
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 6-main.py
    #!/usr/bin/env python3

    from align import FaceAlign
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from utils import load_images, save_images

    fa = FaceAlign('models/landmarks.dat')
    images, filenames = load_images('HBTN', as_array=False)
    anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
    aligned = []
    for image in images:
        aligned.append(fa.align(image, np.array([36, 45, 33]), anchors, 96))
    aligned = np.array(aligned)
    print(aligned.shape)
    if not os.path.isdir('HBTNaligned'):
        print(save_images('HBTNaligned', aligned, filenames))
        os.mkdir('HBTNaligned')
    print(save_images('HBTNaligned', aligned, filenames))
    print(os.listdir('HBTNaligned'))
    image = plt.imread('HBTNaligned/KirenSrinivasan.jpg')
    plt.imshow(image)
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./6-main.py
    False
    True
    ['MariaCoyUlloa4.jpg', 'TuVo0.jpg', 'XimenaCarolinaAndradeVargas1.jpg', 'RodrigoCruz4.jpg', 'LeineValente0.jpg', 'JuanValencia1.jpg', 'DennisPham3.jpg', 'NgaLa3.jpg', 'RodrigoCruz0.jpg', 'LeineValente4.jpg', 'HeimerRojas5.jpg', 'LauraRoudge5.jpg', 'FaizanKhan2.jpg', 'KennethCortesAguas4.jpg', 'AdamSedki3.jpg', 'FaizanKhan1.jpg', 'KennethCortesAguas1.jpg', 'FrancescaCantor6.jpg', 'SamieAzad3.jpg', 'DavidKwan2.jpg', 'DiegoAndrésCastellanosRodríguez0.jpg', 'JulienneTesoro6.jpg', 'PhuTruong2.jpg', 'JohnCook4.jpg', 'RussellMolimock1.jpg', 'SnehaDasaLakshminath3.jpg', 'AnthonyLe1.jpg', 'AndrewMaring3.jpg', 'YesidGonzalez1.jpg', 'HeimerRojas6.jpg', 'RussellMolimock4.jpg', 'DiegoAndrésCastellanosRodríguez1.jpg', 'AdamSedki.jpg', 'NgaLa2.jpg', 'LauraVasquezBernal.jpg', 'CarlosArias5.jpg', 'FrancescaCantor2.jpg', 'ArthurDamm1.jpg', 'DennisPham6.jpg', 'BrentJanski0.jpg', 'XimenaCarolinaAndradeVargas.jpg', 'OlgaLucíaRodríguezToloza0.jpg', 'FaizanKhan0.jpg', 'LeoByeon0.jpg', 'ElaineYeung4.jpg', 'ChristianWilliams0.jpg', 'JohnCook3.jpg', 'KoomeMwiti0.jpg', 'DavidKwan.jpg', 'JuanValencia4.jpg', 'RodrigoCruz1.jpg', 'DavidKwan3.jpg', 'JaiberRamirez3.jpg', 'AnthonyLe2.jpg', 'RodrigoCruz.jpg', 'TimAssavarat2.jpg', 'SamuelAlexanderFlorez7.jpg', 'SamieAzad0.jpg', 'FrancescaCantor4.jpg', 'CarlosArias4.jpg', 'JavierCañon5.jpg', 'GiovannyAlexanderRubioAlbornoz1.jpg', 'HeimerRojas3.jpg', 'RussellMolimock0.jpg', 'NgaLa5.jpg', 'KyleLitscher3.jpg', 'PhuTruong3.jpg', 'AlishaSmith3.jpg', 'JulienneTesoro5.jpg', 'TuVo2.jpg', 'SamieAzad2.jpg', 'AlishaSmith0.jpg', 'SamuelAlexanderFlorez.jpg', 'BrentJanski2.jpg', 'KennethCortesAguas.jpg', 'DennisPham7.jpg', 'XimenaCarolinaAndradeVargas6.jpg', 'ArthurDamm.jpg', 'RyanHudson.jpg', 'YesidGonzalez4.jpg', 'LauraVasquezBernal0.jpg', 'SamieAzad5.jpg', 'DianaBoada6.jpg', 'NgaLa.jpg', 'DiegoAndrésCastellanosRodríguez4.jpg', 'HaroldoVélezLora4.jpg', 'KoomeMwiti4.jpg', 'MiaMorton1.jpg', 'AlishaSmith2.jpg', 'NgaLa4.jpg', 'XimenaCarolinaAndradeVargas0.jpg', 'DianaBoada1.jpg', 'TuVo4.jpg', 'MarkHedgeland1.jpg', 'SofiaCheung2.jpg', 'RicardoGutierrez6.jpg', 'BrentJanski1.jpg', 'CarlosArias0.jpg', 'RodrigoCruz5.jpg', 'TuVo3.jpg', 'LauraVasquezBernal4.jpg', 'XimenaCarolinaAndradeVargas4.jpg', 'BrentJanski6.jpg', 'ElaineYeung3.jpg', 'DianaBoada3.jpg', 'SamieAzad6.jpg', 'SamuelAlexanderFlorez3.jpg', 'RodrigoCruz7.jpg', 'RicardoGutierrez4.jpg', 'HeimerRojas2.jpg', 'KoomeMwiti.jpg', 'FaizanKhan3.jpg', 'JuanValencia3.jpg', 'TimAssavarat.jpg', 'JaiberRamirez1.jpg', 'SnehaDasaLakshminath2.jpg', 'LeineValente2.jpg', 'SamieAzad1.jpg', 'HaroldoVélezLora0.jpg', 'ElaineYeung7.jpg', 'HaroldoVélezLora.jpg', 'CarlosArias.jpg', 'GiovannyAlexanderRubioAlbornoz5.jpg', 'BrentJanski4.jpg', 'HongtuHuang.jpg', 'RodrigoCruz2.jpg', 'HeimerRojas.jpg', 'AlishaSmith.jpg', 'NgaLa0.jpg', 'PhuTruong0.jpg', 'JulienneTesoro2.jpg', 'ArthurDamm2.jpg', 'BrendanEliason4.jpg', 'XimenaCarolinaAndradeVargas5.jpg', 'JaiberRamirez7.jpg', 'AndrewMaring1.jpg', 'MohamethSeck0.jpg', 'AllisonWeiner.jpg', 'KirenSrinivasan0.jpg', 'LeineValente1.jpg', 'DennisPham.jpg', 'SamuelAlexanderFlorez2.jpg', 'AndrewMaring.jpg', 'BrendanEliason3.jpg', 'HeimerRojas1.jpg', 'RicardoGutierrez8.jpg', 'MohamethSeck2.jpg', 'KoomeMwiti3.jpg', 'MohamethSeck6.jpg', 'CarlosArias2.jpg', 'ElaineYeung.jpg', 'JohnCook1.jpg', 'JaiberRamirez4.jpg', 'ElaineYeung6.jpg', 'LeoByeon.jpg', 'BrendanEliason1.jpg', 'RussellMolimock3.jpg', 'KirenSrinivasan.jpg', 'RobertSebastianCastellanosRodriguez.jpg', 'RicardoGutierrez5.jpg', 'RicardoGutierrez3.jpg', 'RicardoGutierrez1.jpg', 'RussellMolimock2.jpg', 'JaiberRamirez.jpg', 'JaiberRamirez2.jpg', 'RicardoGutierrez.jpg', 'DavidLatorre.jpg', 'DianaBoada10.jpg', 'JulienneTesoro7.jpg', 'PhuTruong5.jpg', 'SofiaCheung0.jpg', 'YesidGonzalez3.jpg', 'JavierCañon2.jpg', 'KennethCortesAguas2.jpg', 'JuanValencia.jpg', 'MariaCoyUlloa.jpg', 'LauraVasquezBernal1.jpg', 'DavidKwan1.jpg', 'SnehaDasaLakshminath.jpg', 'BrentJanski7.jpg', 'AndrewMaring2.jpg', 'KyleLitscher0.jpg', 'HaroldoVélezLora1.jpg', 'LeoByeon3.jpg', 'MarkHedgeland2.jpg', 'JuanDavidAmayaGaviria.jpg', 'HeimerRojas7.jpg', 'DianaBoada0.jpg', 'SofiaCheung7.jpg', 'ElaineYeung8.jpg', 'AnthonyLe4.jpg', 'ElaineYeung2.jpg', 'JaiberRamirez5.jpg', 'SamuelAlexanderFlorez6.jpg', 'FaizanKhan.jpg', 'MarkHedgeland.jpg', 'OlgaLucíaRodríguezToloza4.jpg', 'DennisPham5.jpg', 'MiaMorton2.jpg', 'DianaBoada4.jpg', 'DennisPham2.jpg', 'JavierCañon.jpg', 'MohamethSeck3.jpg', 'RodrigoCruz3.jpg', 'PhuTruong1.jpg', 'FeliciaHsieh.jpg', 'JavierCañon4.jpg', 'KyleLitscher2.jpg', 'LauraRoudge2.jpg', 'TimAssavarat4.jpg', 'FrancescaCantor3.jpg', 'OlgaLucíaRodríguezToloza5.jpg', 'HaroldoVélezLora3.jpg', 'KyleLitscher.jpg', 'MiaMorton0.jpg', 'OlgaLucíaRodríguezToloza3.jpg', 'BrentJanski.jpg', 'NgaLa7.jpg', 'AdamSedki1.jpg', 'SamuelAlexanderFlorez1.jpg', 'LauraRoudge1.jpg', 'SofiaCheung.jpg', 'GiovannyAlexanderRubioAlbornoz3.jpg', 'ElaineYeung1.jpg', 'HeimerRojas10.jpg', 'GiovannyAlexanderRubioAlbornoz2.jpg', 'ArthurDamm0.jpg', 'RicardoGutierrez0.jpg', 'MarkHedgeland0.jpg', 'JuanValencia0.jpg', 'LeineValente.jpg', 'LeoByeon1.jpg', 'LauraVasquezBernal3.jpg', 'AlishaSmith4.jpg', 'CarlosArias3.jpg', 'JulienneTesoro0.jpg', 'BrendanEliason0.jpg', 'RodrigoCruz6.jpg', 'MiaMorton3.jpg', 'RobertSebastianCastellanosRodriguez4.jpg', 'MiaMorton4.jpg', 'SamieAzad4.jpg', 'BrendanEliason.jpg', 'AndrewMaring0.jpg', 'LauraRoudge0.jpg', 'PhuTruong.jpg', 'MarianaPlazas.jpg', 'MariaCoyUlloa3.jpg', 'SofiaCheung6.jpg', 'YesidGonzalez0.jpg', 'SofiaCheung3.jpg', 'BrendanEliason2.jpg', 'AlishaSmith1.jpg', 'DianaBoada8.jpg', 'JulienneTesoro3.jpg', 'MarkHedgeland3.jpg', 'JuanValencia2.jpg', 'JaiberRamirez0.jpg', 'DianaBoada.jpg', 'FrancescaCantor1.jpg', 'RodrigoCruz10.jpg', 'SofiaCheung5.jpg', 'ChristianWilliams.jpg', 'DennisPham0.jpg', 'RodrigoCruz8.jpg', 'TimAssavarat0.jpg', 'DavidKwan4.jpg', 'RobertSebastianCastellanosRodriguez0.jpg', 'TuVo1.jpg', 'TuVo.jpg', 'SnehaDasaLakshminath4.jpg', 'TimAssavarat3.jpg', 'RussellMolimock.jpg', 'YesidGonzalez2.jpg', 'HeimerRojas8.jpg', 'XimenaCarolinaAndradeVargas3.jpg', 'RobertSebastianCastellanosRodriguez1.jpg', 'ElaineYeung0.jpg', 'SofiaCheung1.jpg', 'ArthurDamm6.jpg', 'BrentJanski5.jpg', 'YesidGonzalez.jpg', 'FrancescaCantor5.jpg', 'ArthurDamm5.jpg', 'KoomeMwiti1.jpg', 'GiovannyAlexanderRubioAlbornoz7.jpg', 'SamuelAlexanderFlorez0.jpg', 'KennethCortesAguas0.jpg', 'LeineValente3.jpg', 'LauraRoudge3.jpg', 'KyleLitscher1.jpg', 'GiovannyAlexanderRubioAlbornoz4.jpg', 'PhuTruong4.jpg', 'MariaCoyUlloa0.jpg', 'OlgaLucíaRodríguezToloza1.jpg', 'AdamSedki2.jpg', 'OlgaLucíaRodríguezToloza.jpg', 'JohnCook0.jpg', 'MohamethSeck5.jpg', 'SnehaDasaLakshminath1.jpg', 'MiaMorton.jpg', 'CarlosArias6.jpg', 'AdamSedki0.jpg', 'LauraRoudge4.jpg', 'JohnCook2.jpg', 'MohamethSeck4.jpg', 'AndrewMaring4.jpg', 'GiovannyAlexanderRubioAlbornoz6.jpg', 'DianaBoada2.jpg', 'ChristianWilliams3.jpg', 'JavierCañon6.jpg', 'DianaBoada7.jpg', 'MariaCoyUlloa2.jpg', 'ChristianWilliams1.jpg', 'FrancescaCantor.jpg', 'KyleLitscher4.jpg', 'NgaLa6.jpg', 'DiegoAndrésCastellanosRodríguez3.jpg', 'KennethCortesAguas3.jpg', 'GiovannyAlexanderRubioAlbornoz.jpg', 'DavidKwan0.jpg', 'ArthurDamm3.jpg', 'SamuelAlexanderFlorez8.jpg', 'OlgaLucíaRodríguezToloza2.jpg', 'DennisPham1.jpg', 'RicardoGutierrez2.jpg', 'RobertSebastianCastellanosRodriguez2.jpg', 'SofiaCheung4.jpg', 'SamieAzad.jpg', 'SnehaDasaLakshminath0.jpg', 'DiegoAndrésCastellanosRodríguez2.jpg', 'AnthonyLe.jpg', 'ChristianWilliams2.jpg', 'RicardoGutierrez7.jpg', 'NgaLa1.jpg', 'SamuelAlexanderFlorez5.jpg', 'DiegoAndrésCastellanosRodríguez.jpg', 'KirenSrinivasan4.jpg', 'CarlosArias1.jpg', 'BrittneyGoertzen.jpg', 'SamuelAlexanderFlorez4.jpg', 'ArthurDamm4.jpg', 'JaiberRamirez6.jpg', 'AndresMartinPeñaRivera.jpg', 'XimenaCarolinaAndradeVargas2.jpg', 'JavierCañon3.jpg', 'KirenSrinivasan2.jpg', 'DennisPham8.jpg', 'BrentJanski3.jpg', 'FrancescaCantor0.jpg', 'JulienneTesoro1.jpg', 'PhuTruong6.jpg', 'OmarMartínezBermúdez.jpg', 'HeimerRojas4.jpg', 'DianaBoada5.jpg', 'JavierCañon0.jpg', 'LauraVasquezBernal2.jpg', 'DennisPham4.jpg', 'HaroldoVélezLora2.jpg', 'ElaineYeung5.jpg', 'SamuelAlexanderFlorez10.jpg', 'JosefGoodyear.jpg', 'KirenSrinivasan1.jpg', 'MarkHedgeland4.jpg', 'MariaCoyUlloa1.jpg', 'MohamethSeck.jpg', 'GiovannyAlexanderRubioAlbornoz0.jpg', 'AnthonyLe3.jpg', 'LauraRoudge.jpg', 'LeoByeon2.jpg', 'KoomeMwiti2.jpg', 'JulienneTesoro.jpg', 'KirenSrinivasan3.jpg', 'JavierCañon1.jpg', 'ChristianWilliams5.jpg', 'RobertSebastianCastellanosRodriguez3.jpg', 'MariaCoyUlloa5.jpg', 'JulienneTesoro4.jpg', 'HeimerRojas0.jpg', 'NathanPetersen.jpg', 'JohnCook.jpg', 'ChristianWilliams4.jpg', 'AnthonyLe0.jpg', 'MohamethSeck1.jpg', 'TimAssavarat1.jpg']
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x0B-face_verification`
*   File: `utils.py`






#### 7\. Generate Triplets <span class="alert alert-warning mandatory-optional">mandatory</span>

Also in `utils.py`, write a function `def generate_triplets(images, filenames, triplet_names):` that generates triplets:

*   `images` is a `numpy.ndarray` of shape `(n, h, w, 3)` containing the various images in the dataset
*   `filenames` is a list of length `n` containing the corresponding filenames for `images`
*   `triplet_names` is a list of lists where each sublist contains the filenames of an anchor, positive, and negative image, respectively
*   Returns: a list `[A, P, N]`
    *   `A` is a `numpy.ndarray` of shape `(m, h, w, 3)` containing the anchor images for all `m` triplets
    *   `P` is a `numpy.ndarray` of shape `(m, h, w, 3)` containing the positive images for all `m` triplets
    *   `N` is a `numpy.ndarray` of shape `(m, h, w, 3)` containing the negative images for all `m` triplets
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 7-main.py
    #!/usr/bin/env python3

    from utils import load_images, load_csv, generate_triplets
    import numpy as np
    import matplotlib.pyplot as plt

    images, filenames = load_images('HBTNaligned', as_array=True)
    triplet_names = load_csv('FVTriplets.csv')
    A, P, N = generate_triplets(images, filenames, triplet_names)
    plt.subplot(1, 3, 1)
    plt.title('Anchor:' + triplet_names[0][0])
    plt.imshow(A[0])
    plt.subplot(1, 3, 2)
    plt.title('Positive:' + triplet_names[0][1])
    plt.imshow(P[0])
    plt.subplot(1, 3, 3)
    plt.title('Negative:' + triplet_names[0][2])
    plt.imshow(N[0])
    plt.tight_layout()
    plt.show()
    alexa@ubuntu-xenial:0x0B-face_verification$ ./7-main.py
```


**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `supervised_learning/0x0B-face_verification`
*   File: `utils.py`






#### 8\. Initialize Triplet Loss <span class="alert alert-warning mandatory-optional">mandatory</span>

Create a custom layer class `TripletLoss` that inherits from `tensorflow.keras.layers.Layer`:

*   Create the class constructor `def __init__(self, alpha, **kwargs):`
    *   `alpha` is the alpha value used to calculate the triplet loss
    *   sets the public instance attribute `alpha`
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 8-main.py
    #!/usr/bin/env python3

    from triplet_loss import TripletLoss

    print(TripletLoss.__bases__)
    tl = TripletLoss(0.2)
    print(tl.alpha)
    print(sorted(tl.__dict__.keys()))
    alexa@ubuntu-xenial:0x0B-face_verification$  ./8-main.py
    (<class 'tensorflow.python.keras.engine.base_layer.Layer'>,)
    0.2
    ['_activity_regularizer', '_call_convention', '_call_fn_args', '_callable_losses', '_compute_previous_mask', '_dtype', '_dynamic', '_eager_losses', '_expects_training_arg', '_inbound_nodes', '_initial_weights', '_layers', '_losses', '_metrics', '_metrics_tensors', '_mixed_precision_policy', '_name', '_non_trainable_weights', '_obj_reference_counts_dict', '_outbound_nodes', '_self_setattr_tracking', '_trainable_weights', '_updates', 'alpha', 'built', 'input_spec', 'stateful', 'supports_masking', 'trainable']
    alexa@ubuntu-xenial:0x0B-face_verification$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `triplet_loss.py`






#### 9\. Calculate Triplet Loss <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TripletLoss`:

*   Create the public instance method `def triplet_loss(self, inputs):`
    *   `inputs` is a list containing the anchor, positive and negative output tensors from the last layer of the model, respectively
    *   Returns: a tensor containing the triplet loss values
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 9-main.py
    #!/usr/bin/env python3

    import numpy as np
    import tensorflow as tf
    from triplet_loss import TripletLoss

    np.random.seed(0)
    tl = TripletLoss(0.2)
    A = np.random.uniform(0, 1, (2, 128))
    P = np.random.uniform(0, 1, (2, 128))
    N = np.random.uniform(0, 1, (2, 128))

    with tf.Session() as sess:
        loss = tl.triplet_loss([A, P, N])
        print(type(loss))
        print(loss.eval())
    alexa@ubuntu-xenial:0x0B-face_verification$ ./9-main.py
    <class 'tensorflow.python.framework.ops.Tensor'>
    [0\.         3.31159856]
    alexa@ubuntu-xenial:0x0B-face_verification$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `triplet_loss.py`






#### 10\. Call Triplet Loss <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TripletLoss`:

*   Create the public instance method `def call(self, inputs):`
    *   `inputs` is a list containing the anchor, positive, and negative output tensors from the last layer of the model, respectively
    *   adds the triplet loss to the graph
    *   Returns: the triplet loss tensor
```
    alexa@ubuntu-xenial:0x0B-face_verification$ cat 10-main.py
    #!/usr/bin/env python3

    import numpy as np
    import tensorflow as tf
    from triplet_loss import TripletLoss

    np.random.seed(0)
    tl = TripletLoss(0.2)
    A = np.random.uniform(0, 1, (2, 128))
    P = np.random.uniform(0, 1, (2, 128))
    N = np.random.uniform(0, 1, (2, 128))

    inputs = [tf.keras.Input((128,)), tf.keras.Input((128,)), tf.keras.Input((128,))]
    output = tl(inputs)

    model = tf.keras.models.Model(inputs, output)
    print(output)
    print(tl._losses)
    print(model.losses)
    print(model.predict([A, P, N]))
    alexa@ubuntu-xenial:0x0B-face_verification$ ./10-main.py
    Tensor("triplet_loss/Maximum:0", shape=(?,), dtype=float32)
    [<tf.Tensor 'triplet_loss/Maximum:0' shape=(?,) dtype=float32>]
    [<tf.Tensor 'triplet_loss/Maximum:0' shape=(?,) dtype=float32>]
    [0\.       3.311599]
    alexa@ubuntu-xenial:0x0B-face_verification$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `triplet_loss.py`






#### 11\. Initialize Train Model <span class="alert alert-warning mandatory-optional">mandatory</span>

Create the class `TrainModel` that trains a model for face verification using triplet loss:

*   Create the class constructor `def __init__(self, model_path, alpha):`
    *   `model_path` is the path to the base face verification embedding model
        *   loads the model using `with tf.keras.utils.CustomObjectScope({'tf': tf}):`
        *   saves this model as the public instance method `base_model`
    *   `alpha` is the alpha to use for the triplet loss calculation
    *   Creates a new model:
        *   inputs: `[A, P, N]`
            *   `A` is a `numpy.ndarray` containing the anchor images
            *   `P` is a `numpy.ndarray` containing the positive images
            *   `N` is a `numpy.ndarray` containing the negative images
        *   outputs: the triplet losses of `base_model`
        *   compiles the model with `Adam` optimization and no additional losses
        *   save this model as the public instance attribute `training_model`
*   you can use `from triplet_loss import TripletLoss`

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `train_model.py`






#### 12\. Train <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TrainModel`:

*   Create the public instance method `def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3, verbose=True):` that trains `self.training_model`:
    *   `triplets` is a list containing the inputs to `self.training_model`
    *   `epochs` is the number of epochs to train for
    *   `batch_size` is the batch size for training
    *   `validation_split` is the validation split for training
    *   `verbose` is a boolean that sets the verbosity mode
    *   Returns: the `History` output from the training

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `train_model.py`






#### 13\. Save <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TrainModel`:

*   Create the public instance method `def save(self, save_path):` that saves the base embedding model:
    *   `save_path` is the path to save the model
    *   Returns: the saved model

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `train_model.py`






#### 14\. Calculate Metrics <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TrainModel`:

*   static method `def f1_score(y_true, y_pred):`
    *   `y_true`
    *   `y_pred`
    *   Returns: the f1 score
*   static method `def accuracy(y_true, y_pred):`
    *   `y_true`
    *   `y_pred`
    *   Returns: the accuracy

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `train_model.py`






#### 15\. Best Tau <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `TrainModel`:

*   public instance method `def best_tau(self, images, identities, thresholds):`
    *   `images`
    *   `identities`
    *   `thresholds`
    *   Returns: `(tau, f1, acc)`
        *   `tau`
        *   `f1`
        *   `acc`

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `train_model.py`






#### 16\. Initialize Face Verification <span class="alert alert-warning mandatory-optional">mandatory</span>

Create the class `FaceVerification`:

*   class constructor `def __init__(self, model, database, identities):`
    *   `model` is either the fave verification embedding model or the path to where the model is stored
        *   you will need to use `with tf.keras.utils.CustomObjectScope({'tf': tf}):` to load the model
    *   `database` is a `numpy.ndarray` of all the face embeddings in the database
    *   `identities` is a list of identities corresponding to the embeddings in the database
    *   Sets the public instance attributes `database` and `identities`

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `verification.py`






#### 17\. Embedding <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `FaceVerification`:

*   public instance method `def embedding(self, images):`:
    *   `images` are the images to retrieve the embeddings of
    *   Returns: a `numpy.ndarray` of embeddings

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `verification.py`






#### 18\. Verify <span class="alert alert-warning mandatory-optional">mandatory</span>

Update the class `FaceVerification`:

*   public instance method `def verify(self, image, tau=0.5):`:
    *   `image` is the aligned image of the face to be verify
    *   `tau` is the maximum euclidean distance used for verification
    *   Returns: `(identity, distance)`, or `(None, None)` on failure
        *   `identity` is the identity of the verified face
        *   `distance` is the euclidean distance between the verified face embedding and the identified database embedding

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `0x0B-face_verification`
*   File: `verification.py`
