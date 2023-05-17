# PredicemeEsta
Para gestionar las dependencias de python primero hay que crear un nuevo entorno de anaconda y descargar las dependencias:
```
$ conda create -n PredicemeEsta python=3.7
$ conda activate PredicemeEsta
$ chmod +x installer.sh
$ ./installer.sh
```
# Ejecutar crearModelo.py:
## Para Naive-Bayes, tf-idf, Traducir Emojis/Emotes, Clustering:
$ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 2 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,name,retweet_count -o virgin_negative -x on -c t "Virgin America" negative
## Para Logistic Regression, tf-idf, Traducir Emojis/Emotes, Clustering:
$ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 3 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,name,retweet_count -o virgin_negative -x on -c t "Virgin America" negative
 
# Ejecutar probarModelo.py:
$ python probarModelo.py -f TweetsTestParaAlumnos.csv -m virgin_negative -t airline_sentiment -a 3 -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,name,retweet_count -x on -c t "Virgin America" negative
