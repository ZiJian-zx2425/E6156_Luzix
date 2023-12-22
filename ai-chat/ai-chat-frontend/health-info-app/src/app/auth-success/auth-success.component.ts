import {Component, OnInit} from '@angular/core';
import {ActivatedRoute, Router} from "@angular/router";
import {AuthService} from "../services/auth/auth.service";

@Component({
  selector: 'app-auth-success',
  templateUrl: './auth-success.component.html',
  styleUrls: ['./auth-success.component.css']
})
export class AuthSuccessComponent implements OnInit{

  constructor(
    private route: ActivatedRoute,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      const token = params['token'];
      if (token) {
        this.authService.storeToken(token);
        const userRole = this.authService.getUserRole();
        if (userRole === 'doctor') {
          this.router.navigate(['/chat']);
        } else if (userRole === 'patient') {
          // For other roles, navigate to the /schedule route
          this.router.navigate(['/chat']);
        } else if (userRole === 'volunteer') {
          // this.router.navigate(['/appointments']);
          this.router.navigate(['/chat']);
        }
      } else {
        console.error('Authentication token is missing');
        this.router.navigate(['/login']);
      }
    });
  }

}
