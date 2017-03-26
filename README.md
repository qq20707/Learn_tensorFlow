
git init

git commit -m 'xxxxxx'

git remote add origin https://github.com/xxxxx/xxxxx.git

git push origin master

fatal:remote origin already exists　

则执行以下语句：

git remote rm origin


再往后执行git remote add origin https://github.com/xxxxx/xxxxx.git即可。

在执行git push origin master时，报错：

error:failed to push som refs to .....

则执行以下语句：

git pull origin master

# Learn_tensorFlow
TensorFlow深度学习文件

https://github.com/qq20707/Learn_tensorFlow.git
