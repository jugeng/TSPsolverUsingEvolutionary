//let array = [[12,10],[15,25],[4,3],[23,30],[1,10],[7,12],[11 ,18],[14,16],[15,27],[5,26],[25,14],[10,3],[14,4],[18,22],[6,24]]

var array = []
var sent_files = false
var running = false



var stage = new Konva.Stage({
    container: 'map_grid',
    width: 1500,
    height: 2000,
    draggable: true
});

var citylayer = new Konva.Layer();
var routelayer = new Konva.Layer();
stage.add(routelayer)
stage.add(citylayer)

var scaleBy = 1.2;
stage.on('wheel', e => {
    e.evt.preventDefault();
    var oldScale = stage.scaleX();

    var mousePointTo = {
        x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
        y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale
    };

    var newScale =
        e.evt.deltaY > 0 ? oldScale * scaleBy : oldScale / scaleBy;

    stage.scale({
        x: newScale,
        y: newScale
    });

    var newPos = {
        x:
            -(mousePointTo.x - stage.getPointerPosition().x / newScale) *
            newScale,
        y:
            -(mousePointTo.y - stage.getPointerPosition().y / newScale) *
            newScale
    };
    stage.position(newPos);
    stage.batchDraw();
});

depot_pos = 1

var best_route = []


document.getElementById('fileinput').addEventListener('change', readSingleFile, false);

function readSingleFile(evt) {

    if (array.length > 0) {
        reset_array()
    }

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
                            city.push(parseFloat(e))
                        }
                    })
                    array.push(city)
                    add_to_list(city[0].toFixed(0), city[1].toFixed(3), city[2].toFixed(3))
                };
            });
            console.log(array)
            document.getElementById("run").style.display = "block"
            document.getElementById("butn").style.display = "none"
            document.getElementById("job_size").innerHTML = array.length
            document.getElementById('fileinput').value = ""
            draw_cities()




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
    addCity();

    if (array.length > 0) {
        document.getElementById("run").disabled = true
        document.getElementById("run").innerHTML = "Running"
        running = true

        result = eel.runAlgo(0.05)((res) => {
            running = false
            document.getElementById("run").disabled = false
            document.getElementById("run").innerHTML = "Run Algorithm"
            console.log(res)
        })

    } else {
        console.log("No cities added!")
    }
}

eel.expose(update_distance)

function update_distance(val, new_route) {
    console.log(new_route)
    document.getElementById("minimum_distance").innerHTML = val
    best_route = new_route
    route_draw()
}

function show_ex_options(val) {
    if (running === false) {
        x = val.split(" ")
        y = "options " + x[1]
        document.getElementById(y).style.display = "block"
    }



}

function hide_ex_options(val) {
    x = val.split(" ")
    y = "options " + x[1]
    document.getElementById(y).style.display = "none"

}

function reset_depot() {

    return null

}

function make_depot(val) {


    v = "block " + depot_pos
    if (document.getElementById(v) != null) {
        document.getElementById(v).setAttribute("class", "")

    }

    g = val.split(" ")
    v = "block " + g[1]

    depot_pos = g[1]
    document.getElementById(v).setAttribute("class", "warehouse")
    console.log(depot_pos)

    i = 0
    while (i < array.length) {
        if (array[i][0] == depot_pos) {
            k = array.splice(i, 1)
            array.splice(0, 0, k[0])
            show_depot(val)
            break
        }
        i += 1
    }



}

function add_to_list(i, a, b) {
    if (a.length !== 0 && b.length !== 0) {

        let ul = document.getElementById("city_list");

        city = document.createElement("li");
        let t1 = "block " + i;
        city.setAttribute("id", t1);
        city.setAttribute("onmouseover", "show_ex_options(this.id)")
        city.setAttribute("onmouseout", "hide_ex_options(this.id)")
        if (i == depot_pos) {
            city.setAttribute("class", "warehouse");
        }
        div = document.createElement("div");
        div.setAttribute("class", "citycords");


        cityname = document.createElement("h2")
        cityname.style.color = "#104a6b"
        cityname.textContent = "Job " + i

        cityco = document.createElement("p")
        cityco.textContent = "Lat: " + a + "     |       Long: " + b
        cityco.style.color = "#8a8883"


        div.appendChild(cityname)
        div.appendChild(cityco)


        opt = document.createElement("div")
        t2 = "options " + i
        opt.setAttribute("id", t2)
        opt.setAttribute("style", "display:none;")

        deletebtn = document.createElement("button")
        deletebtn.setAttribute("class", "opt_btn")
        deletebtn.innerHTML = "DELETE"
        deletebtn.setAttribute("id", t2)
        deletebtn.setAttribute("onclick", "remove_city(this.id)")
        deletebtn.setAttribute("style", "display:inline-block;")

        makeDepot = document.createElement("button")
        makeDepot.setAttribute("class", "opt_btn")
        makeDepot.innerHTML = "MAKE DEPOT"
        makeDepot.setAttribute("id", t2)
        makeDepot.setAttribute("onclick", "make_depot(this.id)")


        opt.appendChild(makeDepot)
        opt.appendChild(deletebtn)

        city.appendChild(div)
        city.appendChild(opt)

        ul.appendChild(city)
        //city.scrollIntoView()

    }

}


function remove_city(val) {
    let x = val
    x = x.split(" ")
    var list = document.getElementById("city_list");

    t1 = "block " + x[1]
    let i = 0

    if (array.length === 1) {
        reset_array()
    } else {
        while (i < array.length) {
            if (array[i][0] == x[1]) {

                array.splice(i, 1)
                elem = document.getElementById(t1);
                list.removeChild(elem);
                des_city("#city" + x[1])
                document.getElementById("job_size").innerHTML = array.length

                if (x[1] == depot_pos) {
                    let k = "block " + array[0][0];
                    console.log("New depot", k)
                    make_depot(k);

                }
                break
            }
            i += 1

        }
    }

}



function des_city(val) {
    var shape = stage.find(val);
    shape.destroy()
    citylayer.batchDraw()
}


function reset_array() {
    var list = document.getElementById("city_list");

    // As long as <ul> has a child node, remove it
    while (list.hasChildNodes()) {
        list.removeChild(list.firstChild);
    }
    array = []
    citylayer.destroyChildren()
    citylayer.batchDraw()
    routelayer.destroyChildren()
    routelayer.batchDraw()

    depot_pos = 1
    best_route = []
    document.getElementById("run").style.display = "none"
    document.getElementById("butn").style.display = "block"
    document.getElementById("job_size").innerHTML = "-"
    document.getElementById("minimum_distance").innerHTML = "-"

}



function draw_cities() {

    var maxRowX = array.map(function (row) {
        return row[1]
    });

    window.offsetX = (1000 / Math.max.apply(null, maxRowX));

    var maxRowY = array.map(function (row) {
        return row[2]
    });
    window.offsetY = (800 / Math.max.apply(null, maxRowY));

    citylayer.destroyChildren()
    citylayer.batchDraw()

    for (i = 0; i < array.length; i++) {

        var circle = new Konva.Circle({
            x: parseFloat(array[i][1]) * offsetX,
            y: parseFloat(array[i][2]) * offsetY,
            radius: 5,
            fill: '#0B2027',
            id: 'city' + array[i][0]
        });
        citylayer.add(circle)

    }

    var rect = new Konva.Rect({
        x: array[0][1] * offsetX - 6,
        y: array[0][2] * offsetY - 6,
        width: 12,
        height: 12,
        fill: '#FFBC0A',

        id: "depot",

    });

    citylayer.add(rect)

    let newPos = {
        x: array[0][1] * offsetX,
        y: array[0][2] * offsetY
    }
    stage.position(newPos);
    stage.batchDraw();

}

function route_draw() {

    routelayer.destroyChildren()

    for (i = 0; i < best_route.length - 1; i++) {
        let a = best_route[i]
        let b = best_route[i + 1]
        var route = new Konva.Line({
            points: [array[a][1] * offsetX, array[a][2] * offsetY, array[b][1] * offsetX, array[b][2] * offsetY],
            stroke: '#9F9F9C',
            strokeWidth: 1,
            lineCap: 'round',
            lineJoin: 'round'
        })
        routelayer.add(route)
    }
    routelayer.batchDraw()


}

function show_depot(val) {
    x = val.split(" ")
    shape = stage.find("#depot")

    i = 0
    while (i < array.length) {

        if (array[i][0] == x[1]) {
            shape.setAttr('x', array[i][1] * offsetX - 6)
            shape.setAttr('y', array[i][2] * offsetY - 6)
            citylayer.batchDraw()
            break

        }

    }
}