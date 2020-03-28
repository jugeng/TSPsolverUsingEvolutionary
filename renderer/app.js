let array = [[12,10],[15,25],[4,3],[23,30],[1,10],[7,12],[11 ,18],[14,16],[15,27],[5,26],[25,14],[10,3],[14,4],[18,22],[6,24]]

//let array = []

function addCity() {
    console.log(array)
    d = eel.addCity_using_coords(array)()
    
}

function algoRun() {

    if (array.length > 0) {
        init = eel.initialize()()
        document.getElementById("run").disabled = true
        document.getElementById("push_city").disabled = true

        document.getElementById("run").innerHTML = "Running"

        result = eel.runAlgorithm()((res) => {
            document.getElementById("run").disabled = false
            document.getElementById("push_city").disabled = false
            document.getElementById("run").innerHTML = "Run Algorithm"
            console.log(res)
        })

    }
    else {
        console.log("No cities added!")
    }
    
}

eel.expose(update_distance)

function update_distance(val) {
    console.log(val, typeof(val))
    document.getElementById("minimum_distance").innerHTML = val
}

eel.expose(set_progress)

function set_progress (n) {
    document.getElementById("prog_bar").value = n
}

function add() {
    a = document.getElementById("city_x").value
    b = document.getElementById("city_y").value
    console.log(a,b)
    if(a.length !==  0 && b.length !== 0 ) {
        let coord = [a,b]
        array.push(coord)

        let ul = document.getElementById("city_list")

        city = document.createElement("li")
        city.textContent = coord.toString()

        ul.appendChild(city)

    }
   

    document.getElementById("city_x").value = ""
    document.getElementById("city_y").value = ""

    
}