//let array = [[12,10],[15,25],[4,3],[23,30],[1,10],[7,12],[11 ,18],[14,16],[15,27],[5,26],[25,14],[10,3],[14,4],[18,22],[6,24]]

var array = []
document.getElementById('fileinput').addEventListener('change', readSingleFile, false);

function readSingleFile(evt) {
    if (array.length > 0){reset_array()}


    let f = evt.target.files[0];
    if (f) {
        let r = new FileReader();
        r.onload = function (e) {
            var contents = r.result;
            city_coords = contents.split("\n")
            city_coords.forEach(element => {
                    if (element) {
                        let coords = element.split(" ")
                        let city = []
                        coords.forEach(e => {
                            if (e) {
                                city.push(e)
                            }
                        })
                        array.push(city)
                        add_to_list(city[0], city[1]) 
                    }; 
            });
            addCity();
        };
        r.onerror = function (e) {
            alert("Could not load file")
        };
        r.readAsText(f);
    }
}


function addCity() {
    d = eel.addCity_using_coords(array)((v) => {
        console.log(v);
    });
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

    } else {
        console.log("No cities added!")
    }
}

eel.expose(update_distance)
function update_distance(val) {

    document.getElementById("minimum_distance").innerHTML = val
}

eel.expose(set_progress)
function set_progress(n) {
    document.getElementById("prog_bar").value = n
}

function add() {
    a = document.getElementById("city_x").value
    b = document.getElementById("city_y").value
    array.push([a, b])
    add_to_list(a, b)
}

function add_to_list(a, b) {
    if (a.length !== 0 && b.length !== 0) {

        let ul = document.getElementById("city_list")

        city = document.createElement("li")
        let t1 = "block "+ (array.length-1).toString();
        city.setAttribute("id", t1)

        div = document.createElement("div")
        div.setAttribute("id", "citycords") 
        // cityname = document.createElement("h3")
        // cityname.style.color = "#104a6b"
        // cityname.textContent = "City " + array.length
        cityco = document.createElement("p")
        cityco.textContent = "Lat: "+ a + "     |       Long: " + b
        cityco.style.color = "#8a8883"

        deletebtn = document.createElement("button")
        let t2 = "city "+ (array.length-1).toString();
        deletebtn.setAttribute("id", t2)
        deletebtn.setAttribute("class", "delete")
        deletebtn.setAttribute("onclick", "remove_city(this.id)")
    
        //div.appendChild(cityname)
        div.appendChild(cityco)
        div.appendChild(deletebtn)
               
        city.appendChild(div)

        ul.appendChild(city)
        city.scrollIntoView()

    }
    document.getElementById("city_x").value = ""
    document.getElementById("city_y").value = ""
}


function remove_city(val) {
    let x = val
    x = x.split(" ")
    var list = document.getElementById("city_list");
    t1 = "block " + (x[1]).toString();
    elem = document.getElementById(t1);
    list.removeChild(elem);
     array.splice(x[1], 1)
}


function reset_array() {
    var list = document.getElementById("city_list");

    // As long as <ul> has a child node, remove it
    while (list.hasChildNodes()) {
        list.removeChild(list.firstChild);
    }

    array = []
}