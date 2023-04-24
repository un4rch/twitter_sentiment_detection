# PredicemeEsta
Ejecutar crearModelo.py:
$ python crearModelo.py -f TweetsTrainDev.csv -t airline_sentiment -a 3 -s -v tfidf -n text -e tweet_id,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,name,retweet_count,tweet_coord,tweet_created,tweet_location,user_timezone -o modelo.pkl -x on

Ejecutar probarModelo.py:
$ python probarModelo.py -f TweetsTestSubSample.csv -i mode -m modelo.pkl -v tfidf -t airline_sentiment
