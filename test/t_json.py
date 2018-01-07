import json
if __name__ == '__main__':
    # 将python对象test转换json对象
    test = [{"username":"测试","age":16},(2,3),1]
    print(type(test))
    python_to_json = json.dumps(test)
    print(python_to_json)
    print(type(python_to_json))

    # 将json对象转换成python对象
    json_to_python = json.loads(python_to_json)
    print(json_to_python)
    print(json_to_python[0]["age"])
    # print type(json_to_python)