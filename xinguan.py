from datetime import datetime,date,timedelta

#输入今日日期与查询人数
def date_now():
    year,month,day=map(int,input("请输入今天的日期(example:2023-09-09):").split("-"))
    today=datetime(year,month,day)
    num = int(input("请问你要查询几个人:"))
    return today,num

#输入接种针数与时间
def date_input(num):
    all_data=[]
    for i in range(num):
        jz_num=int(input("请输入已经接种了几针:"))
        year,month,day=map(int,input("请输入最近一次接种的日期(example:2023-09-09):").split("-"))
        all_data.append({jz_num:datetime(year,month,day)})
    return all_data

#根据针数选择函数
#利用datetime中strftime函数与timedelta完成日期相加
def case_0(jz_day:datetime,today):
    return {"True":today.strftime("%Y-%m-%d")}

def case_1(jz_day:datetime,today):
    if (today-jz_day).days<30:
        return {"False": (jz_day+timedelta(days=30)).strftime("%Y-%m-%d")}
    return {"True":today.strftime("%Y-%m-%d")}
def case_2(jz_day:datetime,today):
    if (today-jz_day).days<180:
        return {"False": (jz_day+timedelta(days=180)).strftime("%Y-%m-%d")}
    return {"True":today.strftime("%Y-%m-%d")}

def case_3(jz_day:datetime,today):
    return {"False":""}

def switch_fun(jz_num):
    if jz_num==0:
        return case_0
    elif jz_num==1:
        return case_1
    elif jz_num==2:
        return case_2
    elif jz_num==3:
        return case_3


def date_output(data:list,today):
    out_list=[]
    for each_data in data:
        for jz_num in each_data:
            fun=switch_fun(jz_num)
            out_list.append(fun(each_data[jz_num],today))
    print(out_list)


today,num=date_now()
all_data=date_input(num)
date_output(all_data,today)