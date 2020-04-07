    /*

        This file extends the Travel object to approximate
        the solution of the travelling salesman problem by
        simulated annealing.

        The simulated annealing code in this program is based
        upon the C code presented in section 10.9 of "Numerical
        Recipes in C" by Press, et al., Cambridge University
        Press, [1988] 1992, ISBN 978-0-521-43108-8.  That code
        was, in turn, converted from FORTRAN, which is why
        arrays are indexed from 1, not 0.

    */

   "use strict";

   Travel.prototype.solveInit = function() {
       this.tfactr = 0.9;          /* Annealing schedule -- T reduced by
                                      this factor on each step. */
   }

   //  Set up the incremental annealing solution

   Travel.prototype.solveStart = function(goal, rivercost) {
       goodbad = goal;

       /* Calculate cost of crossing river in terms of length from
          original specification in terms of percent of screen */
       rcross = (1.0 / 100.0) * rivercost;

       if (tracing) {
           show("Solving by simulated annealing: " + this.ncity + " cities.");
           if (rcross != 0) {
               show("River crossing cost: " + rivercost + "% of map size.");
           }
       }

       //  Maximum number of paths tried at any temperature
       this.nover = 100 * this.ncity;

       //  Maximum number of successful path changes before continuing
       this.nlimit = 10 * this.ncity;

       this.t = 0.5;               // Initial temperature

       // Calculate length of initial path, wrapping circularly

       this.path = 0.0;
       var i1, i2;
       for (var i = 1; i < this.ncity; i++) {
          i1 = this.iorder[i];
          i2 = this.iorder[i + 1];
          this.path += alen(this.x[i1], this.x[i2], this.y[i1], this.y[i2]);
       }
       i1 = this.iorder[this.ncity];
       i2 = this.iorder[1];
       this.path += alen(this.x[i1], this.x[i2], this.y[i1], this.y[i2]);

       this.de = 0;
       this.n = new Array(7);
   }

   /*  Perform one step of annealing.  Returns true when a
       solution is found.  */

   Travel.prototype.solveStep = function() {
       var nn;

       this.nsucc = 0;
       for (var k = 1; k <= this.nover; k++) {
           do {
               //  Randomly choose the start and end of the segment
               this.n[1] = 1 + (Math.floor(this.ncity * Math.random()));
               this.n[2] = 1 + (Math.floor((this.ncity - 1) * Math.random()));
               if (this.n[2] >= this.n[1]) {
                   this.n[2] = this.n[2] + 1;
               }
               //  Compute number of cities not on the segment
               nn = 1 + ((this.n[1] - this.n[2] + this.ncity - 1) % this.ncity);
           } while (nn < 3);

           /*  Randomly decide whether to try reversing the segment
               or transporting it elsewhere in the path.  */
           if (irbit1()) {
               //  Transport: randomly pick a destination outside the path
               this.n[3] = this.n[2] + (Math.floor(Math.abs(nn - 2) * Math.random())) + 1;
               this.n[3] = 1 + ((this.n[3] - 1) % this.ncity);
               //  Calculate cost of transporting the segment
               var z = transport_cost(this.x, this.y, this.iorder, this.ncity, this.n);
               this.de = z[0];
               this.n = z[1];
               if (metropolis(this.de, this.t)) {
                   this.nsucc++;
                   this.path += this.de;
                   //  Transport the segment
                   this.iorder = transport(this.iorder, this.ncity, this.n);
               }
           } else {
               //  Reversal: calculate cost of reversing the segment
               this.de = reversal_cost(this.x, this.y, this.iorder, this.ncity, this.n);
               if (metropolis(this.de, this.t)) {
                   this.nsucc++;
                   this.path += this.de;
                   //  Reverse the segment
                   this.iorder = reverse(this.iorder, this.ncity, this.n);
               }
           }

           /*  If we've made sufficient successful changes, we're
               done at this temperature.  */
           if (this.nsucc > this.nlimit) {
               break;
           }
       }

       if (tracing) {
           show("Temp = " + ffixed(this.t, 9, 6) +
                "  Cost = " + ffixed(this.path, 9, 6) +
                "  Moves = " + ffixed(this.nsucc, 5, 0));
       }

       this.t *= this.tfactr;      // Reduce temperature

       return this.nsucc == 0;     // Return true if no moves (frozen)
   }

   /*  Show the optimised path on the console log

       We print the path in order of the itinerary,
       showing the city number, its X and Y co-ordinates,
       the computed cost of the edge from that city to
       the next (wrapping around at the bottom) and an
       "R" if river crossing has a nonzero cost and this
       edge crosses the river.  */

   Travel.prototype.showPath = function() {
       show("     City        X         Y       Cost");
       for (var i = 1; i <= this.ncity; i++) {
           var ii = this.iorder[i],
               jj = this.iorder[(i == this.ncity) ? 1 : (i + 1)],
               cost = alen(this.x[ii], this.x[jj],
                           this.y[ii], this.y[jj]),
               crossing =
                   (((rcross != 0) && (this.x[ii] < 0.5) !=
                    (this.x[jj] < 0.5))) ? " R" : "";
           show("     " +
               ffixed(ii, 3, 0) +
               "     " + ffixed(this.x[ii], 8, 4) +
               "  " + ffixed(this.y[ii], 8, 4) +
               " " + ffixed(cost, 8, 4) +
               crossing);
       }
   }

   //  These global variables are hidden arguments to the functions below

   var goodbad = 1;                // Maximise (1) or minimise (-1)
   var rcross = 0;                 // Cost (in length) to cross river
   var showmoves;                  // Show moves ?
   var tracing = false;            // Trace solution ?
   var log;                        // Where to log debug output

   /*  If we're running under node.js, this is the
       main program.

       usage: nodejs travelling.js [cities [goal [river]]]
       where   cities  Number of cities
               goal    1 to minimise, -1 to maximise
               river   Cost to cross river, -100 to 100  */

   if ((typeof window) === "undefined") {
       var t = new Travel();

       var cities = 30, goal = 1, river = 0;

       if (process.argv[2]) {
           cities = parseInt(process.argv[2]);
       }
       if (process.argv[3]) {
           goal = parseInt(process.argv[3]);
       }
       if (process.argv[4]) {
           river = parseInt(process.argv[4]);
       }


       tracing = true;
       t.newProblem(cities);
       t.placeCities();
       if (true) {
           t.solve(goal, river);
       } else {
           t.annealStart(goal, river);
           for (var j = 0; j < (t.ncity * 4); j++) {
               if (t.annealStep()) {
                   break;
               }
           }
       }
       t.showPath();
   }

   /*  Calculate cost function for a proposed path reversal.
       Returns the cost if the path n[1] to n[2] were to be
       reversed.  */

   function reversal_cost(
                          x,      // Array of city X co-ordinates
                          y,      // Array of city Y co-ordinates
                          iorder, // Array containing current itinerary
                          ncity,  // Number of cities
                          n       // Array n[1] = start city n[2] = end city
       ) {
       var xx = new Array(7),
           yy = new Array(7);
       var j, ii, de;

           n[3] = 1 + ((n[1] + ncity - 2) % ncity);
           n[4] = 1 + (n[2] % ncity);
           for (j = 1; j <= 4; j++) {
               ii = iorder[n[j]];
               xx[j] = x[ii];
               yy[j] = y[ii];
           }
           de = -alen(xx[1], xx[3], yy[1], yy[3]) -
                 alen(xx[2], xx[4], yy[2], yy[4]) +
                 alen(xx[1], xx[4], yy[1], yy[4]) +
                 alen(xx[2], xx[3], yy[2], yy[3]);

       return de;
   }

   /*  Reverse a path segment.  Returns the updated iorder
       array with the specified segment reversed.  */

   function reverse(iorder,    // Current itinerary
                    ncity,     // Number of cities
                    n          // Path to be reversed [1] = start, [2] = end
                   ) {
       var nn, j, k, l, itmp;

       nn = (1 + ((n[2] - n[1] + ncity) % ncity)) / 2;
       for (j = 1; j <= nn; j++) {
           k = 1 + ((n[1] + j - 2) % ncity);
           l = 1 + ((n[2] - j + ncity) % ncity);
           itmp = iorder[k];
           iorder[k] = iorder[l];
           iorder[l] = itmp;
       }

       return iorder;
   }

   /*  Compute cost of transport along a given path.  Returns
       an array containing the cost and the updated path
       array in which:
           n[1]    Start of path to be transported
           n[2]    End of path to be transported
           n[3]    City after which path is to be spliced
           n[4]    City before which the path is to be spliced
           n[5]    City preceding the path
           n[6]    City following the path
       The values in the n[] array returned are used by transport()
       below to actually move the path if the decision is made to
       do so.  */

   function transport_cost(x,      // City X co-ordinates
                           y,      // City Y co-ordinates
                           iorder, // Current itinerary
                           ncity,  // Number of cities
                           n       // [1] Path start [2] Path end [3] Destination
                           ) {
       var xx = new Array(7),
           yy = new Array(7);
       var j, ii, de;

       n[4] = 1 + (n[3] % ncity);      // City following n[3]
       n[5] = 1 + ((n[1] + ncity - 2) % ncity); // City preceding n[1]
       n[6] = 1 + (n[2] % ncity);      // City following n[2]

       //  Extract co-ordinates for the six cities involved
       for (j = 1; j <= 6; j++) {
           ii = iorder[n[j]];
           xx[j] = x[ii];
           yy[j] = y[ii];
       }

       /* Calculate the cost of disconnecting the path
          segment from n[1] to n[2], aplicing it in
          between the two adjacent cities n[3] and n[4],
          and connecting n[5] to n[6]. */
       de = -alen(xx[2], xx[6], yy[2], yy[6]) -
             alen(xx[1], xx[5], yy[1], yy[5]) -
             alen(xx[3], xx[4], yy[3], yy[4]) +
             alen(xx[1], xx[3], yy[1], yy[3]) +
             alen(xx[2], xx[4], yy[2], yy[4]) +
             alen(xx[5], xx[6], yy[5], yy[6]);

       return [ de, n ];
   }

   /*  Do the actual city transport.  Returns the updated
       iorder array.  */

   function transport(iorder,  // Current itinerary
                      ncity,   // Number of cities
                      n        // Path indices from transport_cost() above
                     ) {
       var jorder = new Array(ncity + 1);
       var m1, m2, m3, nn, j, jj;

       m1 = 1 + ((n[2] - n[1] + ncity) % ncity);   // Cities from n[1] to n[2]
       m2 = 1 + ((n[5] - n[4] + ncity) % ncity);   // Cities from n[4] to n[5]
       m3 = 1 + ((n[3] - n[6] + ncity) % ncity);   // Cities from n[6] to n[3]

       nn = 1;
       //  Copy the chosen segment
       for (j = 1; j <= m1; j++) {
           jj = 1 + ((j + n[1] - 2) % ncity);
           jorder[nn++] = iorder[jj];
       }
       if (m2 > 0) {
           //  Copy the segment from n[4] to n[5]
           for (j = 1; j <= m2; j++) {
               jj = 1 + ((j + n[4] - 2) % ncity);
               jorder[nn++] = iorder[jj];
           }
       }
       if (m3 > 0) {
           // Copy the segment from n[6] to n[3]
           for (j = 1; j <= m3; j++) {
               jj = 1 + ((j + n[6] - 2) % ncity);
               jorder[nn++] = iorder[jj];
           }
       }

       return jorder;
   }

   /*  Metropolis algorithm.  Returns a Boolean value which
       tells whether to accept a reconfiguration which leads to
       a change de in the objective function e.  If de < 0, the
       change is obviously an improvement, so the return is
       always true.  If de > 0, return true with probability
       e ^ (-de / t) where t is the current temperature in the
       annealing schedule.  */

   function metropolis(de, t) {
       return (de < 0) || (Math.random() < Math.exp(-de / t));
   }

   /*  Combinatorial minimisation by simulated annealing.
       Returns the optimised order array.  */

   Travel.prototype.solveTSP =
       function(x, y, iorder, ncity, maxmin, rivercost, tracing) {
       var path, de, t;
       var nover, nlimit, i1, i2, i, j, k, nsucc, nn;
       var idum;
       var n = new Array(7);

       if (tracing) {
           show("Solving by simulated annealing: " + ncity + " cities.");
       }
       goodbad = maxmin;       // Set maximise / minimise
       showmoves = tracing;
       /* Calculate cost of crossing river in terms of length from
          original specification in terms of percent of screen */
       rcross = (1.0 / 100.0) * rivercost;
       if (tracing && (rcross != 0)) {
           show("River crossing cost: " + rivercost + "% of map size.");
       }

       //  Maximum number of paths tried at any temperature
       nover = 100 * ncity;

       //  Maximum number of successful path changes before continuing
       nlimit = 10 * ncity;
       t = 0.5;                // Initial temperature

       /* Calculate length of initial path */

       path = 0.0;
       for (i = 1; i < ncity; i++) {
          i1 = iorder[i];
          i2 = iorder[i + 1];
          path += alen(x[i1], x[i2], y[i1], y[i2]);
       }
       i1 = iorder[ncity];
       i2 = iorder[1];
       path += alen(x[i1], x[i2], y[i1], y[i2]);

       // Calculate length of initial path, wrapping circularly

       for (j = 1; i <= (ncity * 4); j++) {
           nsucc = 0;
           for (k = 1; k <= nover; k++) {
               do {
                   //  Randomly choose the start and end of the segment
                   n[1] = 1 + (Math.floor(ncity * Math.random()));
                   n[2] = 1 + (Math.floor((ncity - 1) * Math.random()));
                   if (n[2] >= n[1]) {
                       n[2] = n[2] + 1;
                   }
                   //  Compute number of cities not on the segment
                   nn = 1 + ((n[1] - n[2] + ncity - 1) % ncity);
               } while (nn < 3);

               /*  Randomly decide whether to try reversing the segment
                   or transporting it elsewhere in the path.  */
               if (irbit1()) {
                   //  Transport: randomly pick a destination outside the path
                   n[3] = n[2] + (Math.floor(Math.abs(nn - 2) * Math.random())) + 1;
                   n[3] = 1 + ((n[3] - 1) % ncity);
                   //  Calculate cost of transporting the segment
                   var z = transport_cost(x, y, iorder, ncity, n);
                   de = z[0];
                   n = z[1];
                   if (metropolis(de, t)) {
                       nsucc++;
                       path += de;
                       //  Transport the segment
                       iorder = transport(iorder, ncity, n);
                   }
               } else {
                   //  Reversal: calculate cost of reversing the segment
                   de = reversal_cost(x, y, iorder, ncity, n);
                   if (metropolis(de, t)) {
                       nsucc++;
                       path += de;
                       //  Reverse the segment
                       iorder = reverse(iorder, ncity, n);
                   }
               }

               /*  If we've made sufficient successful changes, we're
                   done at this temperature.  */
               if (nsucc > nlimit) {
                   break;
               }
           }
           if (tracing) {
               show("Temp = " + ffixed(t, 9, 6) +
                           "  Cost = " + ffixed(path, 9, 6) +
                           "  Moves = " + ffixed(nsucc, 6, 0));
           }

           t *= this.tfactr;               // Reduce temperature

           //  If we made no moves, consider frozen at the solution
           if (nsucc == 0) {
               break;
           }
       }
       if (tracing) {
           show("Solution with cost " +
               path.toFixed(6) + " at temperature " +
               t.toFixed(6));
       }
       return iorder;
   };
