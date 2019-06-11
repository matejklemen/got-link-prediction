# Kill prediction in Game of Thrones
**Authors**:  Jaka Stavanja, Matej Klemen

This repository contains supporting code for our experiments with link prediction for the
Game of Thrones TV show. The data used:
- a *kills network*, constructed from deaths from first 6 seasons of the show 
(obtained from https://deathtimeline.com/ and additionally cleaned up) - nodes are characters and 
they are connected if one character killed the other,
- a *social network* - nodes are characters and are connected if they appear closely in the books
(used as auxiliary data; obtained from https://github.com/melaniewalsh/sample-social-network-datasets/).

## Running the code
First, make sure that you have installed the packages inside `requirements.txt`.
One way to do this is with the following command.
```
$ pip3 install -r requirements.txt
```

To repeat our calculations, simply run `link_prediction.py`.
```
$ python3 link_prediction.py
```

## Results

| Approach  	                         | AUC (std. dev.)  | Precision (std. dev.) | Recall (std. dev.) |
|:--------------------------------------:|:----------------:|:---------------------:|:------------------:|
| Preferential attachment index          | 0.500 (0.020)    | 0.522 (0.113)         | 0.087 (0.000)      |
| Adamic-Adar index                      | 0.500 (0.000)    | 0.000 (0.000)         | 0.000 (0.000)      |
| Community index                        | 0.500 (0.000)    | 0.000 (0.000)         | 0.000 (0.000)      |
| Alive index                            | **0.863 (0.032)**| 0.822 (0.020)         | 0.930 (0.000)      |
| K-Nearest-Neighbors                    | 0.659 (0.035)    | 0.725 (0.016)         | 0.435 (0.004)      |
| Logistic Regression                    | 0.658 (0.033)    | 0.816 (0.016)         | 0.418 (0.013)      |
| SVM                                    | 0.686 (0.058)    | 0.719 (0.058)         | 0.456 (0.056)      |

## TODO:
- add pdf of paper (after it is "final")
