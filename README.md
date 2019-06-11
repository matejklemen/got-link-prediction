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

## Possible TODO:
- add a results section? or
- add pdf of paper (after it is "final")
