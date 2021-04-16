# 初始化 #
相对一个新项目进行git管理，运行
'git init'
会在工程目录下生产.git文件夹，该目录包含了所有构成git仓库骨架的所有文件。
此时没有对任何文件进行跟踪。
你可以新建一个.gitignore文件，忽略你不想跟踪的文件或文件夹
执行
'git add .'
会暂存除.gitignore里忽略的文件外的所有文件
执行
'git commit -m 'str''
提交到仓库
# 其它命令 #
'git remote'
显示远程仓库
'git config --global --list'
查看配置的用户名和邮箱
'git config --global user.name "yourname"'
'git config --global user.email "youremail"'
配置你的用户名和邮箱
'git remote add alias url'
添加远程仓库