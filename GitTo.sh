#!/bin/bash
git -T git@github.com
git init
git add .


git commit -m "python_learn"
git remote rm origin
git remote add origin https://github.com/qq20707/Learn_tensorFlow.git
git push origin master


