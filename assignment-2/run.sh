#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=./ABAGAIL.jar:$CLASSPATH

# traveling salesman
echo "traveling salesman"
jython travelingsalesman.py

# count ones
echo "count ones"
jython countones.py

# four peaks
echo "four peaks"
jython fourpeaks.py