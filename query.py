import mysql.connector
import streamlit as st

#connection

conn=mysql.connector.connect(
    host="localhost",
    user="angelicahappy",
    password="sidabutar26_",
    database="tugasakhir"
)
c=conn.cursor()

#fetch

def view_all_data():
    c.execute("select * from pasienbaru order by id asc")
    data=c.fetchall()
    return data

def view_all_data_baru():
    c.execute("select * from databaru order by id asc")
    data=c.fetchall()
    return data