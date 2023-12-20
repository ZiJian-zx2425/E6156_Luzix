// role.guard.ts
import { Injectable } from '@angular/core';
import { CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot, UrlTree, Router } from '@angular/router';
import { Observable } from 'rxjs';
import {AuthService} from "./services/auth/auth.service";

@Injectable({
  providedIn: 'root'
})
export class RoleGuard implements CanActivate {
  constructor(private authService: AuthService, private router: Router) {}

  canActivate(): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    const userRole = this.authService.getUserRole();

    if (userRole === 'patient') {
      return true;
    } else if (userRole === 'patient') {
      return this.router.parseUrl('/officehour');  // Use parseUrl to return a UrlTree
    }

    // If none of the above roles match, redirect to a default route or return false
    return this.router.parseUrl('/default-route');  // Replace with your default route
  }
}
