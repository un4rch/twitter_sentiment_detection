# PredicemeEsta
 Para gestionar las dependencias de python primero hay que ejecutar un par de comandos:
 $ chmod +x installer.sh
 $ ./installer.sh

 # Ejecutar crearModelo.py:
  # Para Logistic Regression, tf-idf, Traducir Emojis/Emotes:
    $ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 3 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,name,retweet_count,tweet_coord,tweet_created,tweet_location,user_timezone -o modelo.pkl -x on -c f
  # Para Naive-Bayes, tf-idf, Traducir Emojis/Emotes:
    $ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 2 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,name,retweet_count,tweet_coord,tweet_created,tweet_location,user_timezone -o modelo.pkl -x on -c f
  # Para Clustering:
    $ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 3 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,name,retweet_count,tweet_coord,user_timezone -o modelo.pkl -x on -c t "Virgin America" negative
 # Ejecutar probarModelo.py:
  $ python probarModelo.py -f TweetsTestSubSample.csv -i mode -m modelo.pkl -v tfidf -t airline_sentiment
