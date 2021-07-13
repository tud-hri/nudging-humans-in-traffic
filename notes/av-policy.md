# Notes AV policy

## 250521

See to-do as well

Main decision: go for open-loop experiment first before continuing to optimization implementation

![20210525-notes-meeting-approach](https://user-images.githubusercontent.com/11727203/119492510-72c52e80-bd5f-11eb-81bb-e5cd726baaf3.png)


## 080421

### Implementation of the human's cognitive model in the AV's policy.

**Option 1**: directly / explicitly model parameters in cost function
_human centered_: 'the AV wants to help the human to facilitate his/her decision-making'

![\color{white}
\begin{align*}
J_{t} &= J_{AV} + J_h\\
J_h &= w_p \cdot -\left( p_{\text{turn}}(x_{av}, x_h)-0.5\right)^2 \quad\text{maximize/minimize probability of turn}\\
  &+w_{RT} \cdot RT(x_{av}, x_h) \quad \text{minimize response time}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Ccolor%7Bwhite%7D%0A%5Cbegin%7Balign%2A%7D%0AJ_%7Bt%7D+%26%3D+J_%7BAV%7D+%2B+J_h%5C%5C%0AJ_h+%26%3D+w_p+%5Ccdot+-%5Cleft%28+p_%7B%5Ctext%7Bturn%7D%7D%28x_%7Bav%7D%2C+x_h%29-0.5%5Cright%29%5E2+%5Cquad%5Ctext%7Bmaximize%2Fminimize+probability+of+turn%7D%5C%5C%0A++%26%2Bw_%7BRT%7D+%5Ccdot+RT%28x_%7Bav%7D%2C+x_h%29+%5Cquad+%5Ctext%7Bminimize+response+time%7D%0A%5Cend%7Balign%2A%7D%0A)


**Option 2**: indirect / human is an 'uncertain' dynamic obstacle; the AV accounts for this.
_AV centeres_: 'the AV needs a model the human's decision making to predict the human's trajectory to take into account when planning (e.g. a dynamic obstacle)

![\color{white}
\begin{align*}
J_{t} &= w_v\varphi_V+w_{cl} \varphi_{cl}+w_s\varphi_{\text{shoulder}} \\
&+ p_{\text{turn}}(x_{AV}, x_{h}) \varphi_{\text{obstacle}}(x_{AV}, x_h) \quad \text{human is dynamic obstacle with a probability of turn, and response time (distribution).}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Ccolor%7Bwhite%7D%0A%5Cbegin%7Balign%2A%7D%0AJ_%7Bt%7D+%26%3D+w_v%5Cvarphi_V%2Bw_%7Bcl%7D+%5Cvarphi_%7Bcl%7D%2Bw_s%5Cvarphi_%7B%5Ctext%7Bshoulder%7D%7D+%5C%5C%0A%26%2B+p_%7B%5Ctext%7Bturn%7D%7D%28x_%7BAV%7D%2C+x_%7Bh%7D%29+%5Cvarphi_%7B%5Ctext%7Bobstacle%7D%7D%28x_%7BAV%7D%2C+x_h%29+%5Cquad+%5Ctext%7Bhuman+is+dynamic+obstacle+with+a+probability+of+turn%2C+and+response+time+%28distribution%29.%7D%0A%5Cend%7Balign%2A%7D%0A)

### Hypothesized scenarios
@Arkady

### Human model
![\color{white}
dx = \alpha_x (x(t)-x_\text{crit}) + \alpha_v (v(t)-v_\text{crit}) - \alpha_a (a(t)-a_\text{crit}) + \xi
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dx+%3D+%5Calpha_x+%28x%28t%29-x_%5Ctext%7Bcrit%7D%29+%2B+%5Calpha_v+%28v%28t%29-v_%5Ctext%7Bcrit%7D%29+-+%5Calpha_a+%28a%28t%29-a_%5Ctext%7Bcrit%7D%29+%2B+%5Cxi)

or, equivalently,

![\color{white}
dx = \alpha_x x(t) + \alpha_v v(t) - \alpha_a  a(t) -\theta_\text{crit} + \xi 
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Ccolor%7Bwhite%7D%0A%5Cbegin%7Balign%2A%7D%0AJ_%7Bt%7D+%26%3D+w_v%5Cvarphi_V%2Bw_%7Bcl%7D+%5Cvarphi_%7Bcl%7D%2Bw_s%5Cvarphi_%7B%5Ctext%7Bshoulder%7D%7D+%5C%5C%0A%26%2B+p_%7B%5Ctext%7Bturn%7D%7D%28x_%7BAV%7D%2C+x_%7Bh%7D%29+%5Cvarphi_%7B%5Ctext%7Bobstacle%7D%7D%28x_%7BAV%7D%2C+x_h%29+%5Cquad+%5Ctext%7Bhuman+is+dynamic+obstacle+with+a+probability+of+turn%2C+and+response+time+%28distribution%29.%7D%0A%5Cend%7Balign%2A%7D%0A)
### Whiteboard 'screenshots'
where
![\color{white}
\theta_\text{crit} = \alpha_x x_\text{crit} + \alpha_v v_\text{crit} - \alpha_a  a_\text{crit} + \xi 
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Ccolor%7Bwhite%7D%0A%5Cbegin%7Balign%2A%7D%0AJ_%7Bt%7D+%26%3D+w_v%5Cvarphi_V%2Bw_%7Bcl%7D+%5Cvarphi_%7Bcl%7D%2Bw_s%5Cvarphi_%7B%5Ctext%7Bshoulder%7D%7D+%5C%5C%0A%26%2B+p_%7B%5Ctext%7Bturn%7D%7D%28x_%7BAV%7D%2C+x_%7Bh%7D%29+%5Cvarphi_%7B%5Ctext%7Bobstacle%7D%7D%28x_%7BAV%7D%2C+x_h%29+%5Cquad+%5Ctext%7Bhuman+is+dynamic+obstacle+with+a+probability+of+turn%2C+and+response+time+%28distribution%29.%7D%0A%5Cend%7Balign%2A%7D%0A)

![080421-notes-photo1](https://user-images.githubusercontent.com/11727203/114230331-528d0c00-9979-11eb-8ab6-66b6ad26ad69.jpg)

![080421-notes-photo2](https://user-images.githubusercontent.com/11727203/114230348-56b92980-9979-11eb-99ce-2313d6f89f6f.jpg)


[Link to math equation tool](https://tex-image-link-generator.herokuapp.com/)
