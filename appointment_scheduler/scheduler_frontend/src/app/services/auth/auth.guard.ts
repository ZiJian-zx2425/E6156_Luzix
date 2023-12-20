// src/app/services/auth/auth.guard.ts
import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';
import { AuthService } from './auth.service';
import { map } from 'rxjs/operators';
import {Observable} from "rxjs";

@Injectable({
    providedIn: 'root'
})
export class AuthGuard implements CanActivate {

    constructor(private authService: AuthService, private router: Router) {}

    canActivate(): boolean {
        const isLoggedIn = this.authService.isLoggedIn();
        if (!isLoggedIn) {
            // Redirect to the backend login endpoint
            window.location.href = 'http://127.0.0.1:5001/login';
            return false;
        }
        return true;
    }
}
