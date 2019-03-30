if [ "$1" = "pull" ]; then
	echo git pull 
	git subtree pull -P downSamplingTraining https://github.com/yangcyself/SDPoint.git master	
elif [ "$1" = "push" ]; then
	echo git push
	git subtree push -P downSamplingTraining https://github.com/yangcyself/SDPoint.git master
fi

