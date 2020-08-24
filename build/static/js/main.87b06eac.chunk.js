(this["webpackJsonppharmaceutical-sales-prediction-webapp"]=this["webpackJsonppharmaceutical-sales-prediction-webapp"]||[]).push([[0],{26:function(e,t,a){e.exports=a(54)},31:function(e,t,a){},54:function(e,t,a){"use strict";a.r(t);var r=a(0),n=a.n(r),c=a(23),o=a.n(c),s=(a(31),a(3)),l=a(4),i=a(6),m=a(5),d=a(9),p=a.n(d),u=a(25),h=a(24),f=a(8),E=a.n(f),b=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(){return Object(s.a)(this,a),t.apply(this,arguments)}return Object(l.a)(a,[{key:"render",value:function(){var e=this.props.active,t=E()("nav-link",{active:e});return n.a.createElement("li",{className:"nab-item",role:"presentation"},n.a.createElement("a",{className:t,id:this.props.id+"-tab","data-toggle":"tab",href:"#"+this.props.id,role:"tab","aria-controls":this.props.id+"-tab","aria-selected":"false"},this.props.name))}}]),a}(r.Component),v=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(){return Object(s.a)(this,a),t.apply(this,arguments)}return Object(l.a)(a,[{key:"render",value:function(){var e=this.props.active,t=E()("tab-pane","fade",{active:e,show:e});return n.a.createElement("div",{className:t,id:this.props.id,role:"tabpanel","aria-labelledby":this.props.id+"-tab"},this.props.children)}}]),a}(r.Component),y=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(){return Object(s.a)(this,a),t.apply(this,arguments)}return Object(l.a)(a,[{key:"render",value:function(){var e=this.props.forecast;return n.a.createElement("table",{className:"table table-hover"},n.a.createElement("thead",null,n.a.createElement("tr",null,n.a.createElement("th",{scope:"col"},"Date"),n.a.createElement("th",{scope:"col"},"Trend"),n.a.createElement("th",{scope:"col"},"Lower Sales"),n.a.createElement("th",{scope:"col"},"Sales"),n.a.createElement("th",{scope:"col"},"Upper Sales"),n.a.createElement("th",{scope:"col"},"Weekly"),n.a.createElement("th",{scope:"col"},"Yearly"))),n.a.createElement("tbody",null,e.map((function(e){return n.a.createElement("tr",{key:e.ds},n.a.createElement("th",{scope:"row"},new Date(e.ds).toDateString()),n.a.createElement("td",null,Math.round(e.trend)),n.a.createElement("td",null,Math.round(e.yhat_lower)),n.a.createElement("td",null,Math.round(e.yhat)),n.a.createElement("td",null,Math.round(e.yhat_upper)),n.a.createElement("td",null,Math.round(e.weekly)),n.a.createElement("td",null,Math.round(e.yearly)))}))))}}]),a}(r.Component),g=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(){return Object(s.a)(this,a),t.apply(this,arguments)}return Object(l.a)(a,[{key:"render",value:function(){var e=this.props,t=e.forecast,a=e.name;return n.a.createElement("div",{className:"container pt-3"},n.a.createElement("ul",{className:"nav nav-pills mb-3",id:"pills-tab",role:"tablist"},n.a.createElement(b,{id:a+"dashboard",name:"Dashboard",active:!0}),n.a.createElement(b,{id:a+"salesplot",name:"Sales Plot"}),n.a.createElement(b,{id:a+"trendplot",name:"Trend Plot"}),n.a.createElement(b,{id:a+"holidayplot",name:"Holiday Plot"}),n.a.createElement(b,{id:a+"weeklyplot",name:"Weekly Plot"}),n.a.createElement(b,{id:a+"yearlyplot",name:"Yearly Plot"})),n.a.createElement("div",{className:"tab-content",id:"myTabContent"},n.a.createElement(v,{id:a+"dashboard",active:!0},n.a.createElement(y,{forecast:t})),n.a.createElement(v,{id:a+"salesplot"},n.a.createElement("img",{src:"api/plot/"+a+"/sales"})),n.a.createElement(v,{id:a+"trendplot"},n.a.createElement("img",{src:"api/plot/"+a+"/trend"})),n.a.createElement(v,{id:a+"holidayplot"},n.a.createElement("img",{src:"api/plot/"+a+"/holiday"})),n.a.createElement(v,{id:a+"weeklyplot"},n.a.createElement("img",{src:"api/plot/"+a+"/weekly"})),n.a.createElement(v,{id:a+"yearlyplot"},n.a.createElement("img",{src:"api/plot/"+a+"/yearly"}))))}}]),a}(r.Component),k=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(){return Object(s.a)(this,a),t.apply(this,arguments)}return Object(l.a)(a,[{key:"render",value:function(){var e=this.props,t=e.forecast1,a=e.forecastA,r=e.forecastB,c=e.forecastC,o=e.forecastD;return n.a.createElement("div",{className:"container pt-5"},n.a.createElement("ul",{className:"nav nav-tabs",id:"myTab",role:"tablist"},n.a.createElement(b,{id:"store1",name:"Store1"}),n.a.createElement(b,{id:"storeA",name:"Store Type A"}),n.a.createElement(b,{id:"storeB",name:"Store Type B"}),n.a.createElement(b,{id:"storeC",name:"Store Type C"}),n.a.createElement(b,{id:"storeD",name:"Store Type D"})),n.a.createElement("div",{className:"tab-content",id:"myTabContent"},n.a.createElement(v,{active:!0},n.a.createElement("div",{className:"pt-5"},n.a.createElement("h2",{className:"text-secondary text-center p-4"},"Heroku has a request timeout limit of 30 sec and the model takes more than 3 mins to make a forecast, thus we made a forecast 6 weeks in advance and cached it. It takes maximum of 10 sec to load the forecast. Select Store from navigation tabs to get forecast and descriptive plots. ",n.a.createElement("stron",null,"Store1")," is a single randome store, other options are based on store type."))),n.a.createElement(v,{id:"store1"},n.a.createElement(g,{forecast:t,name:"store1"})),n.a.createElement(v,{id:"storeA"},n.a.createElement(g,{forecast:a,name:"storeA"})),n.a.createElement(v,{id:"storeB"},n.a.createElement(g,{forecast:r,name:"storeB"})),n.a.createElement(v,{id:"storeC"},n.a.createElement(g,{forecast:c,name:"storeC"})),n.a.createElement(v,{id:"storeD"},n.a.createElement(g,{forecast:o,name:"storeD"}))))}}]),a}(r.Component),S=function(){return n.a.createElement("div",{class:"container h-100 p-5 mt-5"},n.a.createElement("div",{class:"row h-100 justify-content-center align-items-center"},n.a.createElement("label",null,n.a.createElement(h.a,{icon:u.a,spin:!0,size:"lg"}),"\xa0Loading...")))},O=function(e){Object(i.a)(a,e);var t=Object(m.a)(a);function a(e){var r;return Object(s.a)(this,a),(r=t.call(this,e))._isMounted=!1,r.predict=function(e){r.setState({isLoading:!0});var t=new FormData;t.append("weeks",e),console.log(e),p()({method:"post",url:"/predict",data:t,headers:{"Content-Type":"multipart/form-data"},timeout:6e5}).then((function(e){return console.log(e.data)})).catch((function(e){return r._isMounted&&r.setState({error:e})}))},r.onSearchSubmit=function(e){var t=r.state.weeksToPredict;r.setState({searchKey:t,view:"loading"}),r.predict(t),e.preventDefault()},r.onSearchChange=function(e){r.setState({weeksToPredict:e.target.value})},r.updateForecast=function(e){r.setState({forecast1:JSON.parse(e.forecast1),forecastA:JSON.parse(e.forecastA),forecastB:JSON.parse(e.forecastB),forecastC:JSON.parse(e.forecastC),forecastD:JSON.parse(e.forecastD)})},r.fetchCache=function(){p()("/api/cached").then((function(e){return r.updateForecast(e.data)})).catch((function(e){return r._isMounted&&r.setState({error:e})}))},r.state={searchKey:"",weeksToPredict:null,forecast1:[],forecastA:[],forecastB:[],forecastC:[],forecastD:[],isLoading:!0},r}return Object(l.a)(a,[{key:"componentDidMount",value:function(){var e=this;this.fetchCache(),setTimeout((function(){e.setState({isLoading:!1})}),5e3)}},{key:"render",value:function(){var e=this.state,t=e.isLoading,a=e.forecast1,r=e.forecastA,c=e.forecastB,o=e.forecastC,s=e.forecastD;return n.a.createElement("div",{className:"App page"},n.a.createElement("nav",{className:"navbar navbar-dark bg-secondary"},n.a.createElement("a",{className:"navbar-brand"},"Team Sukur"),n.a.createElement("form",{className:"form-inline"},n.a.createElement("input",{className:"form-control mr-sm-2",type:"search",placeholder:"Week numbers","aria-label":"Search"}),n.a.createElement("button",{className:"btn btn-outline-light my-2 my-sm-0",type:"submit",disabled:!0},"Predict"))),t?n.a.createElement(S,null):n.a.createElement(k,{forecast1:a,forecastA:r,forecastB:c,forecastC:o,forecastD:s}))}}]),a}(r.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));a(21),a(52),a(53);o.a.render(n.a.createElement(n.a.StrictMode,null,n.a.createElement(O,null)),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))}},[[26,1,2]]]);
//# sourceMappingURL=main.87b06eac.chunk.js.map