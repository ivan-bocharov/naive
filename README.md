Some experiments with Naive Bayes classifier implemented in Python.

In order to install all the requirements, run:
```
pip install -r requirements.txt
```
from the root folder of the repository.

Usage example:
```
python main.py -c 5 -std 5.0 --plot --logging
```

All the command line arguments:
-c Classes number, integer

-s Samples number, integer

-f Features number, integer

-box The box of centers of classes, string (a list of numbers, separated with spaces. Example: 1 2 3 4)

-std Standard deviation of class elements distribution, floating point number

-a Averaging method, string (possible variants are micro, macro and binary)

--logging Logging enabling

--plot Plotting enabling

Help is accessible via -h (or --help) argument.
